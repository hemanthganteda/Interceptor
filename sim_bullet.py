"""Bullet high-speed tail-sitter in Isaac Sim 6.x (PhysX), with the hsquad_dynamics force model.

  /opt/IsaacSim/python.sh sim_bullet.py                      # GUI, full mission (take-off, dash, land)
  /opt/IsaacSim/python.sh sim_bullet.py --headless --speed 60 --log logs/flight.csv
  /opt/IsaacSim/python.sh sim_bullet.py --mode readback      # guide §3.3: PhysX mass properties
  /opt/IsaacSim/python.sh sim_bullet.py --mode drop          # guide §3.3 / ladder step 2: drop test
  /opt/IsaacSim/python.sh sim_bullet.py --mode hover         # ladder step 5-ish: closed-loop hover only

PhysX integrates the rigid bodies (base_link + 4 rotor links, imported from assets/bullet_flu.urdf).
Every aerodynamic / propulsive force comes from hsquad_dynamics and is applied to base_link each
1 kHz physics step (guide §1, §4, §5). Rotor joints are velocity-driven for the visual spin only.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["flight", "hover", "drop", "readback"], default="flight")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--speed", type=float, default=55.0, help="dash speed, m/s (placeholder prop tops out ~59)")
parser.add_argument("--altitude", type=float, default=30.0)
parser.add_argument("--cruise-time", type=float, default=6.0)
parser.add_argument("--wind", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="steady wind, world m/s")
parser.add_argument("--duration", type=float, default=None, help="override sim duration, s")
parser.add_argument("--usd", default=os.path.join(HERE, "assets/usd/bullet_flu/bullet_flu.usda"))
parser.add_argument("--log", default=None, help="CSV log path")
parser.add_argument("--render-hz", type=float, default=60.0)
parser.add_argument("--realtime", action="store_true", help="throttle to wall-clock in GUI mode")
parser.add_argument("--keep-open", action="store_true", help="keep the GUI open after the run")
parser.add_argument("--no-spin", action="store_true", help="do not drive the rotor joints")
parser.add_argument("--capture", default=None, help="directory to save chase-camera frames (PNG)")
parser.add_argument("--capture-hz", type=float, default=15.0)
args, _ = parser.parse_known_args()

# SimulationApp runs Kit with --portable, which writes shader/data caches under <install>/kit. On a
# root-owned install that fails and RTX cannot initialise, so redirect to a user-writable root.
if "--portable-root" not in sys.argv:
    kit_dir = os.path.join(os.environ.get("ISAAC_PATH", "/opt/IsaacSim"), "kit")
    if not os.access(kit_dir, os.W_OK):
        portable = os.path.expanduser("~/.cache/isaacsim-portable")
        os.makedirs(portable, exist_ok=True)
        sys.argv += ["--portable-root", portable]

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless, "width": 1600, "height": 900})

import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
import numpy as np  # noqa: E402
from isaacsim.core.experimental.objects import GroundPlane  # noqa: E402
from isaacsim.core.experimental.prims import Articulation, RigidPrim  # noqa: E402
from isaacsim.core.rendering_manager import RenderingManager, ViewportManager  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

sys.path.insert(0, HERE)
from bullet_control import DashMission, TailsitterController  # noqa: E402
from hsquad_dynamics import BulletDynamics, BulletParams, State  # noqa: E402

ROOT = "/World/bullet"
BASE = ROOT + "/Geometry/base_link"
DT = 1.0 / 1000.0
CTRL_DIV = 4  # 250 Hz controller

# ------------------------------------------------------------------------------------------- utils


def quat_wxyz_to_R(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def to_np(a):
    return a.numpy() if hasattr(a, "numpy") else np.asarray(a)


# ------------------------------------------------------------------------------------------- stage

stage = stage_utils.create_new_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")

sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(3000.0)
UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 25.0, 0.0))
UsdLux.DomeLight.Define(stage, "/World/Sky").CreateIntensityAttr(800.0)
GroundPlane("/World/ground", sizes=[4000.0], colors=[[0.36, 0.40, 0.33]])

# physics scene (guide §4)
SimulationManager.set_physics_sim_device("cpu")  # GPU dynamics off for a single vehicle
scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
scene.CreateGravityMagnitudeAttr().Set(9.81)
px = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
px.CreateTimeStepsPerSecondAttr().Set(int(round(1.0 / DT)))
px.CreateSolverTypeAttr().Set("TGS")
px.CreateEnableCCDAttr().Set(True)
px.CreateEnableStabilizationAttr().Set(False)
px.CreateEnableGPUDynamicsAttr().Set(False)
px.CreateBroadphaseTypeAttr().Set("MBP")

stage_utils.add_reference_to_stage(usd_path=args.usd, path=ROOT, variants=[("Physics", "physx")])

params = BulletParams()
# Spawn standing on the tail stubs, nose up (body x = world z), top facing -x (heading +x).
# Rotation body->world = Ry(-90 deg); base_link origin is the nose apex.
Q_NOSE_UP = np.array([np.cos(-np.pi / 4), 0.0, np.sin(-np.pi / 4), 0.0])
z_nose = 0.685 + 0.01
spawn = np.array([0.0, 0.0, z_nose]) if args.mode != "drop" else np.array([0.0, 0.0, 2.0])
q_spawn = Q_NOSE_UP if args.mode != "drop" else np.array([1.0, 0.0, 0.0, 0.0])
root_x = UsdGeom.Xformable(stage.GetPrimAtPath(ROOT))
root_x.ClearXformOpOrder()
root_x.AddTranslateOp().Set(Gf.Vec3d(*spawn))
root_x.AddOrientOp().Set(Gf.Quatf(*[float(v) for v in q_spawn]))


def tune_physx_prims():
    """Per-prim PhysX settings from guide §4 (velocity clamps, no sleep, CCD, no damping)."""
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(ROOT):
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            is_rotor = "rotor_" in prim.GetName()
            rb.CreateMaxLinearVelocityAttr().Set(1.0e3)  # m/s  (>= 300 m/s, guide §4)
            # deg/s: base >= 100 rad/s; rotor links spin at up to ~1800 rad/s
            rb.CreateMaxAngularVelocityAttr().Set(2.0e5 if is_rotor else 2.0e4)
            rb.CreateSleepThresholdAttr().Set(0.0)
            rb.CreateLinearDampingAttr().Set(0.0)
            rb.CreateAngularDampingAttr().Set(0.0)
            rb.CreateEnableCCDAttr().Set(not is_rotor)
            rb.CreateEnableGyroscopicForcesAttr().Set(True)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            art = PhysxSchema.PhysxArticulationAPI.Apply(prim)
            art.CreateSleepThresholdAttr().Set(0.0)
            art.CreateSolverPositionIterationCountAttr().Set(8)
            art.CreateSolverVelocityIterationCountAttr().Set(2)
            art.CreateEnabledSelfCollisionsAttr().Set(False)
        if prim.IsA(UsdPhysics.RevoluteJoint):
            j = PhysxSchema.PhysxJointAPI.Apply(prim)
            j.CreateMaxJointVelocityAttr().Set(2.0e5)  # deg/s
            j.CreateJointFrictionAttr().Set(0.0)
            if args.no_spin:
                UsdPhysics.DriveAPI.Apply(prim, "angular").CreateDampingAttr().Set(0.0)


tune_physx_prims()
SimulationManager.set_physics_dt(DT)
RenderingManager.set_dt(1.0 / args.render_hz) if hasattr(RenderingManager, "set_dt") else None

robot = Articulation(ROOT)
base = RigidPrim(BASE)

app_utils.play()
RenderingManager.render()

dof_names = robot.dof_names
rotor_dofs = [dof_names.index(f"rotor_{i}_joint") for i in range(4)]
print(f"[bullet] physics engine: {SimulationManager.get_active_physics_engine()}  DOFs: {dof_names}  links: {robot.link_names}")

# ------------------------------------------------------------------------------------------- readback (§3.3)
masses = to_np(base.get_masses()).ravel()
coms = to_np(base.get_coms()[0]).reshape(-1)
inert = to_np(base.get_inertias()).reshape(3, 3)
print("[bullet] PhysX base_link readback")
print(f"   mass    {masses[0]:.4f} kg          (spec 3.3626)")
print(f"   CoM     {np.round(coms, 4)}   (spec [-0.3071 0 0], FLU link frame)")
print(f"   inertia diag {np.round(np.diag(inert), 4)}  off-diag max {np.abs(inert - np.diag(np.diag(inert))).max():.2e}"
      "  (spec 0.0496 0.1031 0.1037)")
ok = (
    abs(masses[0] - 3.3626) / 3.3626 < 1e-3
    and np.allclose(coms, [-0.3071, 0, 0], atol=3e-4)
    and np.allclose(np.diag(inert), [0.0496, 0.1031, 0.1037], rtol=1e-3)
)
print(f"   ladder step 1 (import readback): {'PASS' if ok else 'FAIL'}")
if args.mode == "readback":
    app_utils.stop()
    simulation_app.close()
    sys.exit(0 if ok else 1)

# ------------------------------------------------------------------------------------------- loop setup

dyn = BulletDynamics(params)  # wrench about base_link CG
ctl = TailsitterController(params)
st = State.initial(1)
wind = np.asarray(args.wind, dtype=float)
cg_off = params.cg_total - params.cg_body  # base_link CG -> vehicle CG, body frame

mission = None
if args.mode in ("flight", "hover"):
    p0 = spawn + quat_wxyz_to_R(q_spawn) @ params.cg_total
    mission = DashMission(
        p0, altitude=args.altitude, speed=args.speed if args.mode == "flight" else 0.0,
        cruise_time=args.cruise_time if args.mode == "flight" else 0.0,
        hover_time=3.0 if args.mode == "flight" else 15.0,
    )
duration = args.duration or (mission.t_end if mission else 3.0)

# Direct PhysX tensor views with preallocated CPU buffers: the experimental-API wrappers allocate
# and broadcast on every call, which costs more than the PhysX step itself at 1 kHz.
import warp as wp  # noqa: E402

rb_view = base._physics_rigid_body_view
art_view = robot._physics_articulation_view
idx0 = wp.array([0], dtype=wp.int32, device="cpu")
f_buf = wp.zeros((1, 3), dtype=wp.float32, device="cpu")
m_buf = wp.zeros((1, 3), dtype=wp.float32, device="cpu")
p_buf = wp.zeros((1, 3), dtype=wp.float32, device="cpu")
dof_buf = wp.zeros((1, len(dof_names)), dtype=wp.float32, device="cpu")
f_np, m_np, p_np, dof_np = f_buf.numpy(), m_buf.numpy(), p_buf.numpy(), dof_buf.numpy()

viewport_cam = "/OmniverseKit_Persp"
if args.capture:
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

    os.makedirs(args.capture, exist_ok=True)
    # a multiple of the render interval, so every capture lands on a rendered frame
    capture_every = max(1, int(round(1.0 / (args.render_hz * DT)))) * max(1, int(round(args.render_hz / args.capture_hz)))
trail = []
draw = None
if not args.headless:
    try:
        from isaacsim.util.debug_draw import _debug_draw

        draw = _debug_draw.acquire_debug_draw_interface()
    except Exception as exc:  # optional eye candy
        print(f"[bullet] debug draw unavailable: {exc}")

log_rows = []
LOG_HDR = (
    "t,phase,x,y,z,vx,vy,vz,qw,qx,qy,qz,p,q,r,V,alpha_deg,beta_deg,w0,w1,w2,w3,T0,T1,T2,T3,"
    "thr0,thr1,thr2,thr3,v_pack,i_bat,soc,xr,yr,zr,Fax,Fay,Faz,Max,May,Maz"
)
thr = np.zeros(4)
ref = None
diag = {}
n_steps = int(round(duration / DT))
render_every = max(1, int(round(1.0 / (args.render_hz * DT))))
t_wall0 = time.time()
drop_log = []
cam_eye = None
v_pack = params.cells * 4.2

print(f"[bullet] mode={args.mode} duration={duration:.1f}s  dash speed={args.speed:.1f} m/s  physics {1/DT:.0f} Hz")
prof = dict(read=0.0, control=0.0, dynamics=0.0, apply=0.0, physx=0.0, render=0.0)
tic = time.perf_counter
for k in range(n_steps):
    t = k * DT
    t0 = tic()
    tf = rb_view.get_transforms().numpy()[0].astype(float)  # x y z qx qy qz qw
    vel6 = rb_view.get_velocities().numpy()[0].astype(float)  # world linear (CoM), world angular
    pos_w, quat = tf[:3], tf[[6, 3, 4, 5]]
    lin_w, ang_w = vel6[:3], vel6[3:]
    R = quat_wxyz_to_R(quat / np.linalg.norm(quat))
    w_b = R.T @ ang_w
    # PhysX reports base_link CoM velocity; carry it to the full-vehicle CG
    cg_w = pos_w + R @ params.cg_total
    v_cg = lin_w + np.cross(ang_w, R @ cg_off)

    if args.mode == "drop":
        drop_log.append([t, *(pos_w + R @ params.cg_body), *lin_w, *w_b])
        SimulationManager.step()
        if k % render_every == 0:
            RenderingManager.render()
        continue

    t1 = tic(); prof['read'] += t1 - t0
    st.R_wb[0], st.v_world[0], st.omega_body[0] = R, v_cg, w_b

    if k % CTRL_DIV == 0:
        ref = mission(t)
        phase = mission.phase(t)
        on_ground = cg_w[2] < mission.p0[2] + 0.05 and phase in ("spinup", "done")
        if phase in ("spinup",):
            thr = np.full(4, 0.06)
            ctl.reset()
        elif phase == "done":
            thr = np.zeros(4)
        else:
            thr, diag = ctl.compute(CTRL_DIV * DT, cg_w, v_cg, R, w_b, st.rotor_omega[0], v_pack, ref, wind=wind,
                                    airborne=cg_w[2] > mission.p0[2] + 0.1)

    t2 = tic(); prof['control'] += t2 - t1
    F_b, M_b, out = dyn.step(st, thr[None], wind, DT)
    v_pack = float(out["v_pack"][0])
    t3 = tic(); prof['dynamics'] += t3 - t2
    f_np[0] = R @ F_b[0]
    m_np[0] = R @ M_b[0]
    p_np[0] = pos_w + R @ params.cg_body  # wrench is about the base_link CG
    rb_view.apply_forces_and_torques_at_position(f_buf, m_buf, p_buf, idx0, True)
    if not args.no_spin:
        # visual spin; joint axis is -x_body in the imported USD, rotor_spin is the sense about +x_body
        dof_np[0, rotor_dofs] = -params.rotor_spin * st.rotor_omega[0]
        art_view.set_dof_velocity_targets(dof_buf, idx0)

    t4 = tic(); prof['apply'] += t4 - t3
    SimulationManager.step()

    t5 = tic(); prof['physx'] += t5 - t4
    if k % 10 == 0:
        log_rows.append(
            [t, ["spinup", "climb", "hover", "dash", "cruise", "brake", "land", "done"].index(mission.phase(t)),
             *cg_w, *v_cg, *quat, *w_b, out["V"][0], np.degrees(out["alpha"][0]), np.degrees(out["beta"][0]),
             *st.rotor_omega[0], *out["thrust"][0], *thr, v_pack, out["i_bat"][0], st.soc[0], *ref["p"],
             *out["F_aero"][0], *out["M_aero"][0]]
        )
    if k % 1000 == 0:
        print(
            f"[bullet] t={t:6.2f} {mission.phase(t):7s} pos=({cg_w[0]:7.1f},{cg_w[1]:6.1f},{cg_w[2]:6.1f}) "
            f"V={out['V'][0]:5.1f} m/s ({out['V'][0]*3.6:5.0f} km/h) alpha={np.degrees(out['alpha'][0]):6.1f} "
            f"rpm={np.mean(st.rotor_omega[0])*60/2/np.pi:6.0f} Vpack={v_pack:5.2f} I={out['i_bat'][0]:5.1f} "
            f"err={np.linalg.norm(ref['p']-cg_w):5.2f} m"
        )
    if not np.all(np.isfinite(cg_w)):
        print("[bullet] state became non-finite; stopping")
        break

    if k % render_every == 0:
        if not args.headless or args.capture:
            # chase camera: behind and above the flight path, looking at the vehicle
            vh = v_cg.copy()
            vh[2] = 0.0
            back = -vh / np.linalg.norm(vh) if np.linalg.norm(vh) > 3.0 else np.array([-1.0, 0.0, 0.0])
            eye_goal = cg_w + 6.0 * back + np.array([0.0, -4.0, 2.5])
            cam_eye = eye_goal if cam_eye is None else cam_eye + 0.15 * (eye_goal - cam_eye)
            ViewportManager.set_camera_view(viewport_cam, eye=cam_eye.tolist(), target=cg_w.tolist())
            if draw is not None:
                trail.append(tuple(cg_w))
                if len(trail) > 1:
                    draw.clear_lines()
                    pts = trail[-600:]
                    draw.draw_lines(pts[:-1], pts[1:], [(1.0, 0.45, 0.1, 1.0)] * (len(pts) - 1), [3.0] * (len(pts) - 1))
        tr = tic()
        RenderingManager.render()
        prof['render'] += tic() - tr
        if args.capture and k % capture_every == 0:
            capture_viewport_to_file(get_active_viewport(), os.path.join(args.capture, f"frame_{k // capture_every:05d}.png"))
        if args.realtime and not args.headless:
            lag = t - (time.time() - t_wall0)
            if lag > 0:
                time.sleep(lag)

wall = time.time() - t_wall0
print(f"[bullet] simulated {n_steps*DT:.1f}s in {wall:.1f}s wall ({n_steps*DT/wall:.2f}x real time)")
print("[bullet] loop time per physics step (us): " + ", ".join(f"{k} {1e6*v/max(n_steps,1):.0f}" for k, v in prof.items()))

# ------------------------------------------------------------------------------------------- reports
if args.mode == "drop":
    L = np.array(drop_log)
    # free-fall window, before the lowest point (rotor disc, 0.25 m below the axis) reaches the ground
    i0, i1 = int(0.02 / DT), int(0.5 / DT)
    acc = np.diff(L[i0:i1, 6]) / DT
    rot = np.abs(L[: i1, 7:10]).max()
    print(f"[bullet] drop test: mean vertical accel {acc.mean():.3f} m/s^2 (expect -9.81), max |omega| {rot:.2e} rad/s")
    print(f"   ladder step 2 (drop test): {'PASS' if abs(acc.mean() + 9.81) < 0.02 and rot < 1e-3 else 'FAIL'}")
else:
    L = np.array(log_rows)
    if args.log:
        os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
        np.savetxt(args.log, L, delimiter=",", header=LOG_HDR, comments="", fmt="%.6g")
        print(f"[bullet] log written to {args.log}")
    err = np.linalg.norm(L[:, 2:5] - L[:, 33:36], axis=1)
    air = L[:, 1] >= 1
    print(f"[bullet] max speed {L[:,15].max():.1f} m/s ({L[:,15].max()*3.6:.0f} km/h); "
          f"max tracking error {err[air].max():.2f} m; RMS {np.sqrt(np.mean(err[air]**2)):.2f} m; "
          f"final SOC {L[-1,32]:.3f}; final pos ({L[-1,2]:.1f}, {L[-1,3]:.1f}, {L[-1,4]:.2f})")

if args.keep_open and not args.headless:
    while simulation_app.is_running():
        RenderingManager.render()
app_utils.stop()
simulation_app.close()
