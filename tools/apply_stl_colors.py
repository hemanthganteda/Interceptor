"""Carry the per-facet colours of the CAD STL meshes into a URDF-importer USD asset.

The STLs use the Materialise colour extension: an 80-byte header containing "COLOR=" + RGBA (default
colour) and a 15-bit colour in each facet's attribute word (bits 0-4 red, 5-9 green, 10-14 blue; bit
15 = 0 means the facet uses its own colour, 1 means the default). Isaac Sim's URDF importer reads only
the geometry and binds the URDF <material> instead, so the airframe renders as flat "carbon".

This tool, run after the importer, reads each visual's STL and:
  * adds one material per distinct STL colour to payloads/materials.usda,
  * splits multi-colour meshes into materialBind GeomSubsets (payloads/geometries.usd),
  * binds the subsets (or the whole mesh, if single-colour) in payloads/instances.usda.
It relies on the importer preserving STL facet order (checked: face centroids must match).

Usage (needs Isaac Sim's USD libraries):
  /opt/IsaacSim/python.sh tools/apply_stl_colors.py assets/usd/bullet_flu assets/bullet_flu.urdf
"""

from __future__ import annotations

import os
import struct
import sys
import xml.etree.ElementTree as ET

import numpy as np

PREFIX = "stl_"


def read_stl_colors(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (facet centroids [n,3], facet sRGB colours [n,3] in 0..1)."""
    b = open(path, "rb").read()
    n = struct.unpack("<I", b[80:84])[0]
    d = np.frombuffer(b[84 : 84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    header = b[:80]
    k = header.find(b"COLOR=")
    default = np.array(list(header[k + 6 : k + 9]), float) / 255.0 if k >= 0 else np.array([0.6, 0.6, 0.6])
    a = d["a"].astype(np.int64)
    rgb = np.stack([a & 31, (a >> 5) & 31, (a >> 10) & 31], axis=1) / 31.0
    own = (a & 0x8000) == 0
    rgb[~own] = default
    return d["v"].reshape(-1, 3, 3).mean(1), rgb


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def visual_meshes(urdf_path: str) -> dict[str, str]:
    """URDF visual name -> absolute STL path."""
    out = {}
    base = os.path.dirname(os.path.abspath(urdf_path))
    for vis in ET.parse(urdf_path).getroot().iter("visual"):
        mesh = vis.find("geometry/mesh")
        if mesh is None or not vis.get("name"):
            continue
        fn = mesh.get("filename", "")
        fn = fn[len("file://") :] if fn.startswith("file://") else fn
        fn = fn if os.path.isabs(fn) else os.path.join(base, fn)
        if fn.lower().endswith(".stl") and os.path.isfile(fn):
            out[vis.get("name")] = fn
    return out


def apply_stl_colors(usd_dir: str, urdf_path: str) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

    payloads = os.path.join(usd_dir, "payloads")
    geo_stage = Usd.Stage.Open(os.path.join(payloads, "geometries.usd"))
    mat_stage = Usd.Stage.Open(os.path.join(payloads, "materials.usda"))
    inst_stage = Usd.Stage.Open(os.path.join(payloads, "instances.usda"))
    stls = visual_meshes(urdf_path)

    # mesh name in geometries.usd -> {colour key: face indices}
    groups: dict[str, dict[str, np.ndarray]] = {}
    colours: dict[str, np.ndarray] = {}
    for prim in geo_stage.Traverse():
        if prim.GetTypeName() != "Mesh" or prim.GetName() not in stls:
            continue
        mesh = UsdGeom.Mesh(prim)
        centroids, rgb = read_stl_colors(stls[prim.GetName()])
        pts = np.array(mesh.GetPointsAttr().Get())
        idx = np.array(mesh.GetFaceVertexIndicesAttr().Get())
        fvc = np.array(mesh.GetFaceVertexCountsAttr().Get())
        if len(fvc) != len(centroids) or np.any(fvc != 3):
            raise RuntimeError(f"{prim.GetPath()}: {len(fvc)} faces vs {len(centroids)} STL facets; cannot map colours")
        usd_c = pts[idx.reshape(-1, 3)].mean(1)
        scale = np.abs(pts).max() / max(np.abs(centroids).max(), 1e-12)
        if np.abs(usd_c - centroids * scale).max() > 1e-3 * max(np.abs(usd_c).max(), 1.0):
            raise RuntimeError(f"{prim.GetPath()}: face order differs from {stls[prim.GetName()]}")

        q = np.round(rgb * 255).astype(int)
        keys = np.array([f"{r:02x}{g:02x}{b:02x}" for r, g, b in q])
        groups[prim.GetName()] = {}
        for key in np.unique(keys):
            groups[prim.GetName()][key] = np.nonzero(keys == key)[0]
            colours[key] = q[keys == key][0] / 255.0

        # idempotent: drop subsets from an earlier run
        for child in prim.GetChildren():
            if child.GetName().startswith(PREFIX):
                geo_stage.RemovePrim(child.GetPath())
        if len(groups[prim.GetName()]) > 1:
            for key, faces in groups[prim.GetName()].items():
                UsdGeom.Subset.CreateGeomSubset(
                    mesh, PREFIX + key, UsdGeom.Tokens.face, Vt.IntArray(faces.tolist()), "materialBind"
                )
            UsdGeom.Subset.SetFamilyType(mesh, "materialBind", UsdGeom.Tokens.partition)
        summary = ", ".join(f"#{k} {100*len(v)/len(fvc):.1f}%" for k, v in groups[prim.GetName()].items())
        print(f"[apply_stl_colors] {prim.GetPath()}: {summary}")

    # materials.usda: one UsdPreviewSurface per colour (same parameters the importer uses)
    mroot = mat_stage.GetDefaultPrim() or mat_stage.GetPrimAtPath("/Materials")
    for child in mroot.GetChildren():
        if child.GetName().startswith(PREFIX):
            mat_stage.RemovePrim(child.GetPath())
    for key, c in colours.items():
        mat = UsdShade.Material.Define(mat_stage, mroot.GetPath().AppendChild(PREFIX + key))
        sh = UsdShade.Shader.Define(mat_stage, mat.GetPath().AppendChild("PreviewSurface"))
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*srgb_to_linear(c).tolist()))
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")

    # instances.usda: bind per instance (instances may share a geometry, e.g. motor_1_visual_1)
    mat_file = "./materials.usda"
    for inst in inst_stage.GetPrimAtPath("/Instances").GetChildren():
        refs = inst.GetPrimStack()[0].referenceList.prependedItems
        geo = next((r.primPath.name for r in refs if "geometries" in r.assetPath), None)
        if geo not in groups:
            continue
        mesh_prim = inst.GetChild(geo)
        vm = inst.GetChild("VisualMaterials")
        for child in vm.GetChildren():
            if child.GetName().startswith(PREFIX):
                inst_stage.RemovePrim(child.GetPath())
        bound = {}
        for key in groups[geo]:
            m = inst_stage.DefinePrim(vm.GetPath().AppendChild(PREFIX + key), "Material")
            m.SetInstanceable(True)
            m.GetReferences().AddReference(mat_file, mroot.GetPath().AppendChild(PREFIX + key))
            bound[key] = UsdShade.Material(m)
        if len(groups[geo]) == 1:
            UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(next(iter(bound.values())))
        else:
            for key, m in bound.items():
                sub = inst_stage.OverridePrim(mesh_prim.GetPath().AppendChild(PREFIX + key))
                UsdShade.MaterialBindingAPI.Apply(sub).Bind(m)

    for st in (geo_stage, mat_stage, inst_stage):
        st.GetRootLayer().Save()
    print(f"[apply_stl_colors] {len(colours)} STL colours applied to {usd_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    usd_dir, urdf = (os.path.abspath(p) for p in sys.argv[1:3])
    del sys.argv[1:3]
    kit_dir = os.path.join(os.environ.get("ISAAC_PATH", "/opt/IsaacSim"), "kit")
    if not os.access(kit_dir, os.W_OK):  # root-owned install: keep Kit caches somewhere writable
        portable = os.path.expanduser("~/.cache/isaacsim-portable")
        os.makedirs(portable, exist_ok=True)
        sys.argv += ["--portable-root", portable]
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        apply_stl_colors(usd_dir, urdf)
    finally:
        app.close()
