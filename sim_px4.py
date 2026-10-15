"""Bullet quad tailsitter in Isaac Sim 6.x, flown by PX4 SITL (lockstep, simulator_mavlink).

  /opt/IsaacSim/python.sh sim_px4.py                         # GUI, PX4 VTOL mission: takeoff, transition, dash, back-transition, land
  /opt/IsaacSim/python.sh sim_px4.py --headless --capture logs/px4_frames
  /opt/IsaacSim/python.sh sim_px4.py --headless --max-thrust --distance 3500 --capture logs/px4_frames
  /opt/IsaacSim/python.sh sim_px4.py --openloop 0.55 --duration 3   # plant check without PX4: fixed throttle

Physics split:
  PhysX integrates everything with mass: base_link and the four rotor links (articulation, revolute
  joints), so rotor speed, yaw reaction torque, gyroscopic coupling, contacts and gravity are PhysX's.
  Forces PhysX cannot know come from hsquad_dynamics and are applied where they act, every 1 ms step:
    * each rotor link: thrust T_i along the rotor axis, prop drag torque -s_i Q_i about it
    * each rotor joint: motor torque from the ESC/motor/battery model (at the PhysX rotor speed)
    * base_link: wing + fuselage aerodynamic force and moment (wing at c/4, body at its CP)
PX4 gets IMU / mag / baro / airspeed at 250 Hz and GPS at 10 Hz (HIL_SENSOR, HIL_GPS) and returns
motor commands (HIL_ACTUATOR_CONTROLS). Body frame for PX4 = multicopter (hover) frame of the tailsitter
(x = belly, y = starboard, z = aft).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--duration", type=float, default=240.0, help="max sim time, s")
parser.add_argument("--px4-dir", default=os.path.expanduser("~/PX4-Autopilot"))
parser.add_argument("--no-px4-launch", action="store_true", help="PX4 is started separately (make px4_sitl ...)")
parser.add_argument("--keep-params", action="store_true", help="keep PX4 parameters from the previous run")
parser.add_argument("--altitude", type=float, default=30.0)
parser.add_argument("--speed", type=float, default=40.0, help="mission airspeed in fixed-wing, m/s")
parser.add_argument("--distance", type=float, default=1200.0, help="dash distance east, m")
parser.add_argument("--cg-shift", type=float, default=0.0,
                    help="move the base_link CG forward by this much (m), e.g. relocated battery; PhysX, model and PX4 follow")
parser.add_argument("--max-thrust", action="store_true",
                    help="fixed-wing leg at full throttle (FW_THR_MIN = 1), altitude-priority TECS: finds top speed")
parser.add_argument("--wind", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="steady wind, world ENU m/s")
parser.add_argument("--openloop", type=float, default=None, help="no PX4: hold this throttle on all motors")
parser.add_argument("--usd", default=os.path.join(HERE, "assets/usd/bullet_flu/bullet_flu.usda"))
parser.add_argument("--log", default=os.path.join(HERE, "logs/px4_flight.csv"))
parser.add_argument("--render-hz", type=float, default=30.0)
parser.add_argument("--capture", default=None,
                    help="directory for chase-camera frames (PNG), rendered after the flight from recorded poses")
parser.add_argument("--capture-hz", type=float, default=30.0)
parser.add_argument("--cam-dist", type=float, default=3.0, help="chase-camera distance behind the vehicle, m")
parser.add_argument("--replay-window", type=float, nargs=2, default=None, help="replay only this sim-time range, s")
parser.add_argument("--replay", default=None, help="render --capture frames from a flight log CSV; no PX4, no physics")
args, _ = parser.parse_known_args()

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
from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "px4/pylib"))
from hsquad_dynamics import BulletDynamics, BulletParams, State  # noqa: E402
from pymavlink import mavutil  # noqa: E402

ROOT = "/World/bullet"
DT = 1.0 / 1000.0
SENSOR_DIV = 4  # 250 Hz HIL_SENSOR / lockstep period
GPS_DIV = 100  # 10 Hz
LAT0, LON0, ALT0 = 47.397742, 8.545594, 488.0  # PX4 default SITL home
R_EARTH = 6378137.0
MAG_NED = np.array([0.2178, 0.0099, 0.4280])  # gauss, WMM at LAT0/LON0
G = 9.81

# FLU body -> PX4 tailsitter MC (hover) body: x = belly (ventral), y = starboard, z = aft. A PX4 forward
# transition pitches thrust (-z) toward +x, so +x must be the belly for the airframe to end up upright.
C_MC_FLU = np.array([[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]])
C_NED_ENU = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


def quat_wxyz_to_R(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def R_to_quat_wxyz(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])


# ------------------------------------------------------------------------------------------- stage

stage = stage_utils.create_new_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(3000.0)
UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 25.0, 0.0))
UsdLux.DomeLight.Define(stage, "/World/Sky").CreateIntensityAttr(800.0)
GroundPlane("/World/ground", sizes=[20000.0], colors=[[0.36, 0.40, 0.33]])

SimulationManager.set_physics_sim_device("cpu")
scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
scene.CreateGravityMagnitudeAttr().Set(G)
px = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
px.CreateTimeStepsPerSecondAttr().Set(int(round(1.0 / DT)))
px.CreateSolverTypeAttr().Set("TGS")
px.CreateEnableCCDAttr().Set(True)
px.CreateEnableStabilizationAttr().Set(False)
px.CreateEnableGPUDynamicsAttr().Set(False)
px.CreateBroadphaseTypeAttr().Set("MBP")

stage_utils.add_reference_to_stage(usd_path=args.usd, path=ROOT, variants=[("Physics", "physx")])

params = BulletParams()
params.cg_body = params.cg_body + np.array([args.cg_shift, 0.0, 0.0])  # FLU: +x = forward
# Standing on the tail stubs, nose up, belly facing +x (east): PX4's MC heading (+x_mc = belly) is
# east, and a forward transition pitches the nose toward east with the dorsal side up.
R_SPAWN = np.column_stack([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])  # x_f up, z_f (dorsal) west
q_spawn = R_to_quat_wxyz(R_SPAWN)
spawn = np.array([0.0, 0.0, 0.695])
root_x = UsdGeom.Xformable(stage.GetPrimAtPath(ROOT))
root_x.ClearXformOpOrder()
root_x.AddTranslateOp().Set(Gf.Vec3d(*spawn))
root_x.AddOrientOp().Set(Gf.Quatf(*[float(v) for v in q_spawn]))

base_path, rotor_paths = None, {}
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if not path.startswith(ROOT):
        continue
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        is_rotor = prim.GetName().startswith("rotor_")
        if is_rotor:
            rotor_paths[int(prim.GetName().split("_")[1])] = path
        elif prim.GetName() == "base_link":
            base_path = path
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateMaxLinearVelocityAttr().Set(1.0e3)
        rb.CreateMaxAngularVelocityAttr().Set(4.0e5 if is_rotor else 2.0e4)  # deg/s
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
        j.CreateMaxJointVelocityAttr().Set(4.0e5)  # deg/s
        j.CreateJointFrictionAttr().Set(0.0)
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")  # free joint: torque-driven only
        drive.CreateStiffnessAttr().Set(0.0)
        drive.CreateDampingAttr().Set(0.0)
assert base_path and sorted(rotor_paths) == [0, 1, 2, 3], (base_path, rotor_paths)
_xc = UsdGeom.XformCache()
# base_link pose in the root frame (USD matrices are row-vector, hence the transposes)
T_ROOT_BASE = np.linalg.inv(np.array(_xc.GetLocalToWorldTransform(stage.GetPrimAtPath(ROOT))).T) @ np.array(
    _xc.GetLocalToWorldTransform(stage.GetPrimAtPath(base_path))).T

if args.cg_shift:
    # same mass and inertia about the CG, CG moved forward along the fuselage axis
    UsdPhysics.MassAPI(stage.GetPrimAtPath(base_path)).GetCenterOfMassAttr().Set(Gf.Vec3f(*params.cg_body.tolist()))
    print(f"[px4sim] CG shifted {args.cg_shift*1000:+.0f} mm: base_link CoM x = {params.cg_body[0]:.4f}, vehicle CG x = {params.cg_total[0]:.4f}")
SimulationManager.set_physics_dt(DT)
robot = Articulation(ROOT)
base = RigidPrim(base_path)
rotors = RigidPrim([rotor_paths[i] for i in range(4)])
app_utils.play()
RenderingManager.render()

dof_names = robot.dof_names
rotor_dofs = [dof_names.index(f"rotor_{i}_joint") for i in range(4)]
print(f"[px4sim] DOFs {dof_names}; base {base_path}; rotors {[rotor_paths[i] for i in range(4)]}")

import warp as wp  # noqa: E402

rb_view = base._physics_rigid_body_view
rot_view = rotors._physics_rigid_body_view
art_view = robot._physics_articulation_view
idx0 = wp.array([0], dtype=wp.int32, device="cpu")
idx4 = wp.array([0, 1, 2, 3], dtype=wp.int32, device="cpu")
bf, bm, bp = (wp.zeros((1, 3), dtype=wp.float32, device="cpu") for _ in range(3))
rf, rm, rp = (wp.zeros((4, 3), dtype=wp.float32, device="cpu") for _ in range(3))
eff = wp.zeros((1, len(dof_names)), dtype=wp.float32, device="cpu")
bf_np, bm_np, bp_np, rf_np, rm_np, rp_np, eff_np = (a.numpy() for a in (bf, bm, bp, rf, rm, rp, eff))

dyn = BulletDynamics(params)  # aero moment about the base_link CG
st = State.initial(1)
spin = params.rotor_spin
wind = np.asarray(args.wind, dtype=float)
rng = np.random.default_rng(0)

# ------------------------------------------------------------------------------------------- PX4 link

status = dict(vtol="?", landed="?", seq=-1, mode="?", armed=False, done=False, text=[])
px4_proc = None
mav = None
if args.openloop is None and args.replay is None:
    mav = mavutil.mavlink_connection("tcpin:0.0.0.0:4560", source_system=1, source_component=51)
    if not args.no_px4_launch:
        build = os.path.join(args.px4_dir, "build/px4_sitl_default")
        if not args.keep_params:
            for f in ("parameters.bson", "parameters_backup.bson"):
                p = os.path.join(build, "rootfs", f)
                if os.path.exists(p):
                    os.remove(p)
        os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
        env = dict(os.environ, PX4_SYS_AUTOSTART="22001", PX4_SIM_MODEL="isaac_bullet")
        px4_log = open(os.path.join(HERE, "logs/px4_console.log"), "w")
        px4_proc = subprocess.Popen([os.path.join(build, "bin/px4"), "-d"], cwd=build, env=env,
                                    stdout=px4_log, stderr=subprocess.STDOUT)
        print(f"[px4sim] PX4 SITL started (pid {px4_proc.pid}), console -> logs/px4_console.log")
    t_wait = time.time()
    while mav.port is None:
        mav.recv_match(blocking=False)
        if time.time() - t_wait > 60:
            raise RuntimeError("PX4 did not connect on TCP 4560")
        time.sleep(0.01)
    print("[px4sim] PX4 connected on TCP 4560")


def lat_lon(p_enu):
    lat = LAT0 + np.degrees(p_enu[1] / R_EARTH)
    lon = LON0 + np.degrees(p_enu[0] / (R_EARTH * np.cos(np.radians(LAT0))))
    return lat, lon


def mission_thread():
    """Upload and fly: takeoff (MC) -> transition FW -> dash east -> transition MC -> land."""
    gcs = mavutil.mavlink_connection("udpin:0.0.0.0:14540", source_system=255, source_component=190)
    gcs.wait_heartbeat()
    tgt = (gcs.target_system, gcs.target_component)
    px4_params = []
    if args.cg_shift:
        # rotor positions relative to the vehicle CG (MC frame z = aft): 0.2352 m at the nominal CG
        pz = float(-params.rotor_pos[0, 0] + params.cg_total[0])
        px4_params += [(f"CA_ROTOR{i}_PZ", pz) for i in range(4)]
    if args.max_thrust:
        px4_params += [("FW_THR_MIN", 1.0), ("FW_THR_MAX", 1.0), ("FW_AIRSPD_MAX", 150.0), ("FW_T_SPDWEIGHT", 0.0)]
    if px4_params:
        # full throttle in fixed-wing; airspeed target out of reach; TECS holds altitude with pitch only
        for name, val in px4_params:
            for _ in range(10):
                gcs.mav.param_set_send(*tgt, name.encode(), val, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
                m = gcs.recv_match(type="PARAM_VALUE", blocking=True, timeout=1.0)
                if m is not None and m.param_id == name and abs(m.param_value - val) < 1e-3:
                    print(f"[px4sim] {name} = {m.param_value:g}")
                    break
            else:
                print(f"[px4sim] could not set {name}")

    def pump(timeout=0.2):
        m = gcs.recv_match(blocking=True, timeout=timeout)
        if m is None:
            return None
        t = m.get_type()
        if t == "STATUSTEXT":
            print(f"[PX4] {m.text}")
            status["text"].append(m.text)
        elif t == "EXTENDED_SYS_STATE":
            status["vtol"] = {0: "?", 1: "TRANS_FW", 2: "TRANS_MC", 3: "MC", 4: "FW"}.get(m.vtol_state, m.vtol_state)
            status["landed"] = {0: "?", 1: "GROUND", 2: "AIR", 3: "TAKEOFF", 4: "LANDING"}.get(m.landed_state, m.landed_state)
        elif t == "MISSION_CURRENT":
            status["seq"] = m.seq
        elif t == "HEARTBEAT" and m.get_srcComponent() == 1:
            status["armed"] = bool(m.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            status["mode"] = gcs.flightmode
        return m

    def command(cmd, *p, timeout=5.0):
        p = list(p) + [0] * (7 - len(p))
        gcs.mav.command_long_send(*tgt, cmd, 0, *p)
        t0 = time.time()
        while time.time() - t0 < timeout:
            m = pump()
            if m is not None and m.get_type() == "COMMAND_ACK" and m.command == cmd:
                return m.result
        return None

    # mission
    D = args.distance
    wp_lat, wp_lon = lat_lon(np.array([D, 0.0]))
    ld_lat, ld_lon = lat_lon(np.array([D + 250.0, 0.0]))
    alt = args.altitude
    MI = mavutil.mavlink
    F = MI.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT
    items = [
        (MI.MAV_CMD_NAV_TAKEOFF, F, [0, 0, 0, float("nan")], LAT0, LON0, alt),
        (MI.MAV_CMD_DO_VTOL_TRANSITION, MI.MAV_FRAME_MISSION, [MI.MAV_VTOL_STATE_FW, 0, 0, 0], 0, 0, 0),
        (MI.MAV_CMD_DO_CHANGE_SPEED, MI.MAV_FRAME_MISSION, [0, 150.0 if args.max_thrust else args.speed, -1, 0], 0, 0, 0),
        (MI.MAV_CMD_NAV_WAYPOINT, F, [0, 0, 0, float("nan")], wp_lat, wp_lon, alt),
        (MI.MAV_CMD_DO_VTOL_TRANSITION, MI.MAV_FRAME_MISSION, [MI.MAV_VTOL_STATE_MC, 0, 0, 0], 0, 0, 0),
        (MI.MAV_CMD_NAV_LAND, F, [0, 0, 0, float("nan")], ld_lat, ld_lon, 0),
    ]
    for _ in range(5):
        gcs.mav.mission_count_send(*tgt, len(items), MI.MAV_MISSION_TYPE_MISSION)
        t0, ok = time.time(), False
        while time.time() - t0 < 10:
            m = pump()
            if m is None:
                continue
            if m.get_type() in ("MISSION_REQUEST_INT", "MISSION_REQUEST"):
                c, fr, pp, la, lo, al = items[m.seq]
                gcs.mav.mission_item_int_send(*tgt, m.seq, fr, c, int(m.seq == 0), 1, *pp,
                                              int(la * 1e7), int(lo * 1e7), al, MI.MAV_MISSION_TYPE_MISSION)
            elif m.get_type() == "MISSION_ACK":
                ok = m.type == MI.MAV_MISSION_ACCEPTED
                break
        if ok:
            print(f"[px4sim] mission uploaded ({len(items)} items)")
            break
    else:
        print("[px4sim] mission upload failed")
        return

    # arm once preflight checks pass (EKF needs a few seconds of GPS)
    while not status["armed"]:
        r = command(MI.MAV_CMD_COMPONENT_ARM_DISARM, 1)
        if r == MI.MAV_RESULT_ACCEPTED:
            break
        t0 = time.time()
        while time.time() - t0 < 2.0:
            pump()
    print("[px4sim] armed; starting mission")
    command(MI.MAV_CMD_MISSION_START, 0, len(items) - 1)
    was_air = False
    while not status["done"]:
        pump()
        was_air |= status["landed"] == "AIR"
        if was_air and status["landed"] == "GROUND" and not status["armed"]:
            status["done"] = True


sim_now = 0.0  # sim time of the main loop, read by the watchdog


def airspeed_watchdog():
    """Warn if PX4's airspeed_selector stops publishing.

    PX4 v1.16 SITL can stop running its interval-scheduled modules (airspeed_selector,
    battery_simulator) around arming; without validated airspeed a tailsitter never completes
    its front transition and quad-chutes. A restart does not recover it, so this only reports.
    """
    bindir = os.path.join(args.px4_dir, "build/px4_sitl_default/bin")
    warned = False
    while not status["done"] and not warned:
        time.sleep(2.0)
        if sim_now < 6.0:
            continue
        try:
            out = subprocess.run([os.path.join(bindir, "px4-listener"), "airspeed_validated"], capture_output=True,
                                 text=True, timeout=5).stdout
            ts = [int(l.split(":")[1].split()[0]) for l in out.splitlines() if l.strip().startswith("timestamp:")]
        except Exception:
            continue
        if status["done"]:
            break
        if not ts or sim_now - ts[0] / 1e6 > 1.5:
            print(f"[px4sim] WARNING: PX4 airspeed_selector stalled (last output {ts[0] / 1e6 if ts else 'never'} s, "
                  f"now {sim_now:.1f} s); the front transition would time out. Aborting (exit 3) so the run can be retried.")
            status["aspd_stall"] = True
            warned = True


if mav is not None:
    threading.Thread(target=mission_thread, daemon=True).start()
    if not args.no_px4_launch:
        threading.Thread(target=airspeed_watchdog, daemon=True).start()


def chase_camera(cg, v, state):
    """Chase-camera eye position: locked to the vehicle (no positional lag at speed); only the
    viewing direction is smoothed. state is a dict carrying the smoothed direction."""
    vh = np.array([v[0], v[1], 0.0])
    back_goal = -vh / np.linalg.norm(vh) if np.linalg.norm(vh) > 3.0 else np.array([-1.0, 0.0, 0.0])
    back = state.get("back", back_goal)
    back = back + 0.1 * (back_goal - back)
    back /= np.linalg.norm(back)
    state["back"] = back
    side = np.cross(np.array([0.0, 0.0, 1.0]), back)  # to the left of the flight direction
    return cg + args.cam_dist * (back + 0.5 * side + np.array([0.0, 0.0, 0.33]))


def send_sensors(t, R, w_b, a_w, pos, v_w, v_air_b):
    """HIL_SENSOR in the PX4 tailsitter MC body frame."""
    Rt = R.T
    acc = C_MC_FLU @ (Rt @ (a_w - np.array([0.0, 0.0, -G]))) + rng.normal(0, 0.03, 3)
    gyr = C_MC_FLU @ w_b + rng.normal(0, 0.002, 3)
    mag = C_MC_FLU @ (Rt @ (C_NED_ENU.T @ MAG_NED)) + rng.normal(0, 0.002, 3)
    h = ALT0 + pos[2]
    p_hpa = 1013.25 * (1 - 2.25577e-5 * h) ** 5.25588 + rng.normal(0, 0.01)
    vx = max(v_air_b[0], 0.0)  # pitot along the nose
    dp = 0.5 * params.rho * vx * vx / 100.0 + abs(rng.normal(0, 0.005))
    mav.mav.hil_sensor_send(int(t * 1e6) + 1, *acc, *gyr, *mag, p_hpa, dp, h, 15.0 - 0.0065 * pos[2], 0x1FFF, 0)
    return acc


def send_gps(t, pos, v_w):
    lat, lon = lat_lon(pos)
    vn, ve, vd = v_w[1], v_w[0], -v_w[2]
    gs = np.hypot(vn, ve)
    cog = (np.degrees(np.arctan2(ve, vn)) % 360.0) * 100 if gs > 0.5 else 65535
    mav.mav.hil_gps_send(int(t * 1e6) + 1, 3, int(lat * 1e7), int(lon * 1e7), int((ALT0 + pos[2]) * 1000), 50, 80,
                         int(gs * 100), int(vn * 100), int(ve * 100), int(vd * 100), int(cog), 12)


px4_replying = False


def wait_actuators():
    """Lockstep: PX4 answers each HIL_SENSOR with HIL_ACTUATOR_CONTROLS.

    While PX4 boots it does not answer yet, so wait briefly. Once it answers, wait for every
    reply (strict lockstep): advancing sim time while PX4 is stalled, e.g. by the logger opening
    its file at arming, desynchronises PX4's scheduler.
    """
    global px4_replying
    timeout = 10.0 if px4_replying else 0.5
    t0 = time.time()
    while time.time() - t0 < timeout:
        m = mav.recv_match(type="HIL_ACTUATOR_CONTROLS", blocking=True, timeout=timeout)
        if m is not None:
            px4_replying = True
            return np.clip(np.array(m.controls[:4], dtype=float), 0.0, 1.0)
    return None


# ------------------------------------------------------------------------------------------- loop

viewport_cam = "/OmniverseKit_Persp"
frames = []  # (t, base_link world 4x4, cg, velocity) at capture rate, rendered after the flight
render_every = max(1, int(round(1.0 / (args.render_hz * DT))))
capture_every = render_every * max(1, int(round(args.render_hz / args.capture_hz)))
draw = None
if not args.headless and not args.capture:
    try:
        from isaacsim.util.debug_draw import _debug_draw

        draw = _debug_draw.acquire_debug_draw_interface()
    except Exception as exc:
        print(f"[px4sim] debug draw unavailable: {exc}")

thr = np.full(4, args.openloop) if args.openloop is not None else np.zeros(4)
v_prev = None
a_w = np.zeros(3)
cam_eye, cam_state = None, {}
trail, rows, lockstep_misses = [], [], 0
t_wall0 = time.time()
t_done = None
n_steps = 0 if args.replay else int(round(args.duration / DT))
t = 0.0
print(f"[px4sim] physics {1/DT:.0f} Hz, sensors {1/(SENSOR_DIV*DT):.0f} Hz, max {args.duration:.0f} s sim")

for k in range(n_steps):
    t = k * DT
    sim_now = t
    tf = rb_view.get_transforms().numpy()[0].astype(float)  # x y z qx qy qz qw
    vel6 = rb_view.get_velocities().numpy()[0].astype(float)
    pos_w, quat = tf[:3], tf[[6, 3, 4, 5]]
    R = quat_wxyz_to_R(quat / np.linalg.norm(quat))
    lin_w, ang_w = vel6[:3], vel6[3:]
    w_b = R.T @ ang_w
    cg_w = pos_w + R @ params.cg_body
    v_air_b = R.T @ (lin_w - wind)

    if mav is not None and k % SENSOR_DIV == 0:
        if v_prev is not None:
            a_w = (lin_w - v_prev) / (SENSOR_DIV * DT)
        v_prev = lin_w.copy()
        send_sensors(t, R, w_b, a_w, cg_w, lin_w, v_air_b)
        if k % GPS_DIV == 0:
            send_gps(t, cg_w, lin_w)
        u = wait_actuators()
        if u is None:
            lockstep_misses += 1
            if px4_replying:
                print(f"[px4sim] WARNING: no actuator reply from PX4 within 10 s at t={t:.3f}")
        else:
            thr = u

    # rotor state from PhysX
    dof_v = art_view.get_dof_velocities().numpy()[0].astype(float)
    omega = -spin * dof_v[rotor_dofs]  # joint axis is -x_body; spin is the sense about +x_body
    rtf = rot_view.get_transforms().numpy().astype(float)
    rvel = rot_view.get_velocities().numpy().astype(float)
    a_w_axis = R @ params.rotor_axis
    v_axial = (rvel[:, :3] - wind) @ a_w_axis

    tau_m, el = dyn.motor_electrical(st, thr[None], np.maximum(omega, 0.0)[None], DT)
    T, Q, _ = dyn.rotor_loads(np.maximum(omega, 0.0), v_axial)
    F_aero, M_aero, aux = dyn.aero(v_air_b[None], w_b[None])

    rf_np[:] = T[:, None] * a_w_axis[None]
    rm_np[:] = (-spin * Q)[:, None] * a_w_axis[None]
    rp_np[:] = rtf[:, :3]
    rot_view.apply_forces_and_torques_at_position(rf, rm, rp, idx4, True)
    bf_np[0] = R @ F_aero[0]
    bm_np[0] = R @ M_aero[0]
    bp_np[0] = cg_w
    rb_view.apply_forces_and_torques_at_position(bf, bm, bp, idx0, True)
    eff_np[0, :] = 0.0
    eff_np[0, rotor_dofs] = -spin * tau_m[0]
    art_view.set_dof_actuation_forces(eff, idx0)

    SimulationManager.step()

    if k % 20 == 0:
        rows.append([t, *cg_w, *lin_w, *quat, *w_b, aux["V"][0], np.degrees(aux["alpha"][0]), *omega, *thr, *T,
                     el["v_pack"][0], el["i_bat"][0], st.soc[0], {"MC": 0, "TRANS_FW": 1, "FW": 2, "TRANS_MC": 3}.get(status["vtol"], -1)])
    if k % 1000 == 0:
        print(f"[px4sim] t={t:6.1f} {status['mode']:>14s} vtol={status['vtol']:8s} {status['landed']:7s} wp={status['seq']:2d} "
              f"pos=({cg_w[0]:7.1f},{cg_w[1]:6.1f},{cg_w[2]:6.1f}) V={aux['V'][0]:5.1f} m/s "
              f"rpm={np.mean(omega)*60/2/np.pi:6.0f} thr={np.round(thr, 2)} soc={st.soc[0]:.3f}")
    if not np.all(np.isfinite(cg_w)):
        print("[px4sim] state became non-finite; stopping")
        break

    if args.capture and k % capture_every == 0:
        Tb = np.eye(4)
        Tb[:3, :3], Tb[:3, 3] = R, pos_w
        frames.append((t, Tb, cg_w.copy(), lin_w.copy()))
    if k % render_every == 0 and not args.headless and not args.capture:
        cam_eye = chase_camera(cg_w, lin_w, cam_state)
        ViewportManager.set_camera_view(viewport_cam, eye=cam_eye.tolist(), target=cg_w.tolist())
        if draw is not None:
            trail.append(tuple(cg_w))
            if len(trail) > 1:
                draw.clear_lines()
                pts = trail[-900:]
                draw.draw_lines(pts[:-1], pts[1:], [(1.0, 0.45, 0.1, 1.0)] * (len(pts) - 1), [3.0] * (len(pts) - 1))
        RenderingManager.render()

    if status.get("aspd_stall"):
        break
    if status["done"] and t_done is None:
        t_done = t
        print(f"[px4sim] mission complete, landed and disarmed at t={t:.1f}s")
    if t_done is not None and t > t_done + 3.0:
        break

status["done"] = True  # flight loop over: stops the airspeed watchdog before PX4 is shut down
if rows:
    wall = time.time() - t_wall0
    print(f"[px4sim] simulated {t:.1f}s in {wall:.1f}s wall ({t/wall:.2f}x real time); lockstep timeouts {lockstep_misses}")
    L = np.array(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    np.savetxt(args.log, L, delimiter=",", fmt="%.6g", comments="",
               header="t,x,y,z,vx,vy,vz,qw,qx,qy,qz,p,q,r,V,alpha_deg,w0,w1,w2,w3,thr0,thr1,thr2,thr3,T0,T1,T2,T3,"
                      "v_pack,i_bat,soc,vtol")
    print(f"[px4sim] max airspeed {L[:, 14].max():.1f} m/s ({L[:, 14].max()*3.6:.0f} km/h); max alt {L[:, 3].max():.1f} m; "
          f"final pos ({L[-1, 1]:.1f}, {L[-1, 2]:.1f}, {L[-1, 3]:.2f}); log {args.log}")
if px4_proc is not None:
    px4_proc.terminate()
    try:
        px4_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        px4_proc.kill()
if args.replay:
    # rows: t, cg xyz, v xyz, base quat wxyz, ... (vehicle CG = base_link CG here)
    Lr = np.genfromtxt(args.replay, delimiter=",", names=True)
    t_a, t_b = args.replay_window or (Lr["t"][0], Lr["t"][-1])
    for tc in np.arange(t_a, t_b, 1.0 / args.capture_hz):
        i = min(np.searchsorted(Lr["t"], tc), len(Lr) - 1)
        j = max(i - 1, 0)
        a = 0.0 if i == j else (tc - Lr["t"][j]) / (Lr["t"][i] - Lr["t"][j])
        lerp = lambda keys: (1 - a) * np.array([Lr[k][j] for k in keys]) + a * np.array([Lr[k][i] for k in keys])
        cg, v = lerp(("x", "y", "z")), lerp(("vx", "vy", "vz"))
        q = lerp(("qw", "qx", "qy", "qz"))
        Rr = quat_wxyz_to_R(q / np.linalg.norm(q))
        Tb = np.eye(4)
        Tb[:3, :3], Tb[:3, 3] = Rr, cg - Rr @ params.cg_body
        frames.append((tc, Tb, cg, v))

if status.get("aspd_stall"):
    sys.stdout.flush()
    os._exit(3)  # PX4 already stopped; run_mission.sh retries
if args.capture and frames and not status.get("aspd_stall"):
    # Replay: physics stopped, the recorded base_link poses are written to the root prim and rendered.
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

    app_utils.stop()
    os.makedirs(args.capture, exist_ok=True)
    ops = {op.GetOpType(): op for op in root_x.GetOrderedXformOps()}
    op_t, op_q = ops[UsdGeom.XformOp.TypeTranslate], ops[UsdGeom.XformOp.TypeOrient]
    QuatT = Gf.Quatd if op_q.GetPrecision() == UsdGeom.XformOp.PrecisionDouble else Gf.Quatf
    inv_rb = np.linalg.inv(T_ROOT_BASE)
    vp = get_active_viewport()
    cam_state = {}
    # Chase camera as a USD prim, moved by the same xform writes as the vehicle, so the two cannot
    # drift apart at speed (the viewport's own camera updates through a different, laggier path).
    rig = UsdGeom.Xform.Define(stage, "/World/ChaseRig")
    rig_t = rig.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    cam = UsdGeom.Camera.Define(stage, "/World/ChaseRig/Cam")
    cam.CreateFocalLengthAttr(18.0)
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1.0e5))
    cam_t = cam.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    cam_q = cam.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    vp.camera_path = "/World/ChaseRig/Cam"
    print(f"[px4sim] rendering {len(frames)} frames to {args.capture}")
    for i, (t, Tb, cg, v) in enumerate(frames):
        Tr = Tb @ inv_rb
        op_t.Set(Gf.Vec3d(*Tr[:3, 3].tolist()))
        op_q.Set(QuatT(*R_to_quat_wxyz(Tr[:3, :3]).tolist()))
        cam_eye = chase_camera(cg, v, cam_state)
        zc = cam_eye - cg
        zc /= np.linalg.norm(zc)  # USD cameras look down -Z
        xc = np.cross([0.0, 0.0, 1.0], zc)
        xc /= np.linalg.norm(xc)
        rig_t.Set(Gf.Vec3d(*cg.tolist()))
        cam_t.Set(Gf.Vec3d(*(cam_eye - cg).tolist()))
        cam_q.Set(Gf.Quatd(*R_to_quat_wxyz(np.column_stack([xc, np.cross(zc, xc), zc])).tolist()))
        # RTX shows a moved prim a few renders late (about 2 m behind at 65 m/s with 2 renders):
        # settle 6 renders per frame, then render once more so the async capture sees this state
        for _ in range(12 if i == 0 else 6):
            RenderingManager.render()
        capture_viewport_to_file(vp, os.path.join(args.capture, f"frame_{i:05d}.png"))
        RenderingManager.render()
        if i % 300 == 0:
            print(f"[px4sim]   frame {i}/{len(frames)}")
    for _ in range(5):
        RenderingManager.render()
else:
    app_utils.stop()
simulation_app.close()
