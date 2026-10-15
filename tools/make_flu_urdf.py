#!/usr/bin/env python3
"""Generate the Isaac-ready FLU URDF from the CAD-frame Bullet URDF.

Guide §2 / §3.1:
  * 180 deg yaw about z: x_flu = -x_urdf, y_flu = -y_urdf, z_flu = z_urdf
    (applied to every <origin>, <axis> and inertia tensor).
  * <gazebo> blocks stripped (their values live in hsquad_dynamics).
  * package:// mesh paths resolved against --mesh-dir. If a mesh file is missing,
    the visual is replaced by a primitive proxy so the model still imports.

Usage:
  python3 tools/make_flu_urdf.py high-speed-drone.urdf assets/bullet_flu.urdf [--mesh-dir DIR]
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

RZ_PI = np.diag([-1.0, -1.0, 1.0])  # 180 deg yaw


def rpy_to_mat(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def mat_to_rpy(m: np.ndarray) -> tuple[float, float, float]:
    p = math.asin(max(-1.0, min(1.0, -m[2, 0])))
    if abs(math.cos(p)) > 1e-9:
        r = math.atan2(m[2, 1], m[2, 2])
        y = math.atan2(m[1, 0], m[0, 0])
    else:  # gimbal lock
        r = 0.0
        y = math.atan2(-m[0, 1], m[1, 1])
    return r, p, y


def fmt(v) -> str:
    return " ".join(f"{0.0 if abs(x) < 1e-12 else x:.9g}" for x in v)


def convert_origin(el: ET.Element, is_link_root_frame: bool) -> None:
    """Re-express an <origin> in the yawed frame.

    Every frame (link frames and the visual/collision/inertial frames inside them) is yawed by
    the same Rz(pi), so a transform T_parent_child becomes Rz T Rz^T.
    """
    xyz = np.array([float(s) for s in el.get("xyz", "0 0 0").split()])
    rpy = [float(s) for s in el.get("rpy", "0 0 0").split()]
    xyz_new = RZ_PI @ xyz
    rot_new = RZ_PI @ rpy_to_mat(*rpy) @ RZ_PI.T
    el.set("xyz", fmt(xyz_new))
    el.set("rpy", fmt(mat_to_rpy(rot_new)))


def convert_mesh_origin(el: ET.Element) -> None:
    """Re-express a mesh <origin> in the yawed frame.

    Mesh vertices stay in CAD coordinates, so the mesh frame is not yawed with the link:
    T_link_mesh becomes Rz T (not Rz T Rz^T as for symmetric primitives and child frames).
    """
    xyz = np.array([float(s) for s in el.get("xyz", "0 0 0").split()])
    rpy = [float(s) for s in el.get("rpy", "0 0 0").split()]
    el.set("xyz", fmt(RZ_PI @ xyz))
    el.set("rpy", fmt(mat_to_rpy(RZ_PI @ rpy_to_mat(*rpy))))


def convert_inertia(el: ET.Element) -> None:
    keys = ["ixx", "ixy", "ixz", "iyy", "iyz", "izz"]
    v = {k: float(el.get(k, "0")) for k in keys}
    inertia = np.array(
        [[v["ixx"], v["ixy"], v["ixz"]], [v["ixy"], v["iyy"], v["iyz"]], [v["ixz"], v["iyz"], v["izz"]]]
    )
    inertia = RZ_PI @ inertia @ RZ_PI.T
    for k, (i, j) in zip(keys, [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]):
        el.set(k, f"{inertia[i, j]:.9e}")


# Primitive stand-ins for the CAD meshes (sizes from the collision set / spec sheet).
PROXY_GEOMETRY = {
    "motor": ("cylinder", {"radius": "0.0175", "length": "0.040"}, "0 1.570796327 0", "-0.020 0 0"),
    # two-blade bar (not a disc) so the visual spin is visible
    "propeller": ("box", {"size": "0.006 0.2275 0.022"}, "0 0 0", "0 0 0"),
}


def proxy_for(mesh_name: str):
    for key, spec in PROXY_GEOMETRY.items():
        if mesh_name.startswith(key):
            return spec
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--mesh-dir", default=None, help="directory that replaces package://bullet_sim/meshes")
    args = ap.parse_args()

    tree = ET.parse(args.src)
    robot = tree.getroot()

    # 1. strip <gazebo>
    for g in robot.findall("gazebo"):
        robot.remove(g)

    # 2. drop body_flu (base_link IS the FLU frame after conversion; origin stays at the nose apex)
    for j in robot.findall("joint"):
        if j.get("name") == "body_flu_joint":
            robot.remove(j)
    for l in robot.findall("link"):
        if l.get("name") == "body_flu":
            robot.remove(l)

    # 3. frames
    def mesh_resolves(el: ET.Element) -> bool:
        mesh = el.find("geometry/mesh")
        if mesh is None or not args.mesh_dir:
            return False
        rel = mesh.get("filename", "").split("package://bullet_sim/meshes/")[-1]
        return os.path.isfile(os.path.join(args.mesh_dir, rel))

    for link in robot.findall("link"):
        for child in list(link):
            o = child.find("origin")
            if o is None and child.tag in ("visual", "collision") and mesh_resolves(child):
                o = ET.SubElement(child, "origin", xyz="0 0 0", rpy="0 0 0")
            if o is not None:
                if child.tag in ("visual", "collision") and mesh_resolves(child):
                    convert_mesh_origin(o)
                else:
                    convert_origin(o, False)
            if child.tag == "inertial":
                convert_inertia(child.find("inertia"))
    for joint in robot.findall("joint"):
        o = joint.find("origin")
        if o is not None:
            convert_origin(o, True)
        a = joint.find("axis")
        if a is not None:
            a.set("xyz", fmt(RZ_PI @ np.array([float(s) for s in a.get("xyz").split()])))

    # 4. meshes -> resolved path or primitive proxy
    n_proxy = 0
    for link in robot.findall("link"):
        for vis in link.findall("visual"):
            geom = vis.find("geometry")
            mesh = geom.find("mesh") if geom is not None else None
            if mesh is None:
                continue
            fname = mesh.get("filename")
            rel = fname.split("package://bullet_sim/meshes/")[-1]
            path = os.path.abspath(os.path.join(args.mesh_dir, rel)) if args.mesh_dir else None
            if path and os.path.isfile(path):
                # relative to the output URDF, so the repository works wherever it is checked out
                mesh.set("filename", os.path.relpath(path, os.path.dirname(os.path.abspath(args.dst))))
                continue
            base = os.path.splitext(os.path.basename(rel))[0]
            spec = proxy_for(base)
            geom.remove(mesh)
            if spec is None:
                # airframe: show the collision primitives of this link as the visual
                link.remove(vis)
                for col in link.findall("collision"):
                    v = ET.SubElement(link, "visual", {"name": col.get("name").replace("col_", "vis_")})
                    v.append(copy.deepcopy(col.find("origin")))
                    v.append(copy.deepcopy(col.find("geometry")))
                    ET.SubElement(v, "material", {"name": "carbon"})
            else:
                kind, attrs, rpy, off = spec
                ET.SubElement(geom, kind, attrs)
                o = vis.find("origin")
                # proxies are axis-symmetric about x; place them on the visual's origin, x-aligned
                xyz = np.array([float(s) for s in o.get("xyz").split()]) + np.array([float(s) for s in off.split()])
                o.set("xyz", fmt(xyz))
                o.set("rpy", rpy)
            n_proxy += 1

    ET.indent(tree, space="  ")
    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    with open(args.dst, "wb") as f:
        f.write(b'<?xml version="1.0"?>\n<!-- AUTO-GENERATED by tools/make_flu_urdf.py from ')
        f.write(os.path.basename(args.src).encode())
        f.write(b". FLU frame (x fwd, y port, z up), origin at nose apex. Do not edit. -->\n")
        tree.write(f, encoding="utf-8", xml_declaration=False)
    print(f"wrote {args.dst} ({n_proxy} mesh visuals replaced by proxies)")


if __name__ == "__main__":
    main()
