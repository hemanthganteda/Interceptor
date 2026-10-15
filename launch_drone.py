"""Launch high-speed-drone.urdf in Isaac Sim (plain PhysX, no aero/thrust model).

  /opt/IsaacSim/python.sh launch_drone.py                     # GUI: import URDF, drop drone onto the ground
  /opt/IsaacSim/python.sh launch_drone.py --fix-base          # GUI: drone pinned in the air (inspect the model)
  /opt/IsaacSim/python.sh launch_drone.py --headless --duration 3

The URDF's visual meshes (package://bullet_sim/meshes/...) are not shipped with this repo. Any visual whose
mesh cannot be found is replaced by the link's collision primitives so the airframe is still visible. The
patched URDF and the converted USD are written to build/high_speed_drone/.
"""

from __future__ import annotations

import argparse
import copy
import os
import shutil
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", default=os.path.join(HERE, "high-speed-drone.urdf"))
parser.add_argument("--out", default=os.path.join(HERE, "build/high_speed_drone"))
parser.add_argument("--mesh-dir", default=None, help="directory that holds the bullet_sim package (meshes/visual/*.stl)")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--height", type=float, default=1.0, help="spawn height of the nose apex, m")
parser.add_argument("--fix-base", action="store_true", help="pin base_link to the world")
parser.add_argument("--duration", type=float, default=None, help="stop after this many sim seconds")
args, _ = parser.parse_known_args()

# Root-owned install: Kit's --portable caches must go somewhere writable (same fix as sim_bullet.py).
if "--portable-root" not in sys.argv:
    kit_dir = os.path.join(os.environ.get("ISAAC_PATH", "/opt/IsaacSim"), "kit")
    if not os.access(kit_dir, os.W_OK):
        portable = os.path.expanduser("~/.cache/isaacsim-portable")
        os.makedirs(portable, exist_ok=True)
        sys.argv += ["--portable-root", portable]

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless, "width": 1600, "height": 900})

import omni.kit.app  # noqa: E402

ext_manager = omni.kit.app.get_app().get_extension_manager()
for ext in ("isaacsim.asset.importer.urdf", "isaacsim.robot.schema", "omni.scene.optimizer.core"):
    ext_manager.set_extension_enabled_immediate(ext, True)

import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig  # noqa: E402
from isaacsim.core.experimental.objects import GroundPlane  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

# ------------------------------------------------------------------------------------- URDF prep


def resolve_mesh(filename: str) -> str | None:
    if filename.startswith("package://"):
        rel = filename[len("package://") :].split("/", 1)[1]
        roots = [args.mesh_dir] if args.mesh_dir else []
        roots += [os.path.dirname(args.urdf)]
        for root in roots:
            path = os.path.join(root, rel)
            if os.path.isfile(path):
                return path
        return None
    path = filename[len("file://") :] if filename.startswith("file://") else filename
    path = path if os.path.isabs(path) else os.path.join(os.path.dirname(args.urdf), path)
    return path if os.path.isfile(path) else None


def patch_urdf(src: str, dst: str) -> None:
    tree = ET.parse(src)
    robot = tree.getroot()
    for gz in robot.findall("gazebo"):  # Gazebo-only plugins; the importer has no use for them
        robot.remove(gz)
    for link in robot.findall("link"):
        missing = False
        for vis in link.findall("visual"):
            mesh = vis.find("geometry/mesh")
            if mesh is None:
                continue
            found = resolve_mesh(mesh.get("filename", ""))
            if found:
                mesh.set("filename", found)
            else:
                link.remove(vis)
                missing = True
        if missing:
            mat = "rotor_dark" if link.get("name", "").startswith("rotor_") else "carbon"
            for i, col in enumerate(link.findall("collision")):
                vis = ET.SubElement(link, "visual", name=f"{col.get('name', 'col')}_vis{i}")
                for child in col:
                    vis.append(copy.deepcopy(child))
                ET.SubElement(vis, "material", name=mat)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tree.write(dst, xml_declaration=True, encoding="utf-8")


patched = os.path.join(args.out, "high-speed-drone.urdf")
patch_urdf(args.urdf, patched)

shutil.rmtree(os.path.join(args.out, "usd"), ignore_errors=True)  # importer suffixes _1, _2... otherwise
cfg = URDFImporterConfig(
    urdf_path=patched,
    usd_path=os.path.join(args.out, "usd"),
    fix_base=args.fix_base,
    merge_fixed_joints=False,
    allow_self_collision=False,
)
usd_file = URDFImporter(cfg).import_urdf()
print(f"[launch_drone] URDF -> USD: {usd_file}")

# the importer drops the STLs' per-facet colours; put them back
sys.path.insert(0, os.path.join(HERE, "tools"))
from apply_stl_colors import apply_stl_colors  # noqa: E402

apply_stl_colors(os.path.dirname(usd_file), patched)

# ------------------------------------------------------------------------------------- stage

stage = stage_utils.create_new_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")

sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(3000.0)
UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 25.0, 0.0))
UsdLux.DomeLight.Define(stage, "/World/Sky").CreateIntensityAttr(800.0)
GroundPlane("/World/ground", sizes=[200.0], colors=[[0.36, 0.40, 0.33]])

scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
scene.CreateGravityMagnitudeAttr().Set(9.81)
SimulationManager.set_physics_sim_device("cpu")

ROOT = "/World/high_speed_drone"
src_layer = Usd.Stage.Open(usd_file)
variants = [("Physics", "physx")] if "Physics" in src_layer.GetDefaultPrim().GetVariantSets().GetNames() else None
stage_utils.add_reference_to_stage(usd_path=usd_file, path=ROOT, variants=variants)
xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROOT))
xf.ClearXformOpOrder()
xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, args.height))

if not args.headless:
    from omni.kit.viewport.utility import get_active_viewport
    from omni.kit.viewport.utility.camera_state import ViewportCameraState

    ViewportCameraState("/OmniverseKit_Persp", get_active_viewport()).set_position_world(
        Gf.Vec3d(1.6, -1.8, args.height + 0.8), True
    )
    ViewportCameraState("/OmniverseKit_Persp", get_active_viewport()).set_target_world(
        Gf.Vec3d(0.3, 0.0, max(args.height - 0.3, 0.1)), True
    )

# ------------------------------------------------------------------------------------- run

app_utils.play()
dt = 1.0 / 60.0
t = 0.0
frame = 0
base_path = next(
    (str(p.GetPath()) for p in stage.Traverse() if p.GetName() == "base_link" and p.HasAPI(UsdPhysics.RigidBodyAPI)),
    None,
)
print(f"[launch_drone] playing; base_link prim: {base_path}")

while simulation_app.is_running():
    simulation_app.update()
    t += dt
    frame += 1
    if base_path and frame % 60 == 0:
        pos = UsdGeom.Xformable(stage.GetPrimAtPath(base_path)).ComputeLocalToWorldTransform(0).ExtractTranslation()
        print(f"[launch_drone] t={t:5.2f}s  base_link z={pos[2]:.3f} m")
    if args.duration is not None and t >= args.duration:
        break

app_utils.stop()
simulation_app.close()
