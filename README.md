# Interceptor
A demo interceptor

"Bullet": a 3.4 kg quad tailsitter (four fixed rotors on the tail, thrust along the fuselage, one
straight wing) simulated in NVIDIA Isaac Sim 6.1 (PhysX) and flown by PX4 v1.16 SITL.

## Layout

| Path | What it is |
|---|---|
| `high-speed-drone.urdf` | Source URDF (CAD frame: x aft, y starboard, z up). Revision notes in its header. |
| `meshes/` | CAD meshes (STL, mm, per-facet colours) |
| `assets/bullet_flu.urdf`, `assets/usd/bullet_flu/` | FLU-frame URDF and the Isaac Sim USD asset generated from it |
| `hsquad_dynamics/` | Force model: rotors (C_T/C_Q vs advance ratio), motor/ESC/battery, wing and fuselage aero. Parameters and placeholders in `params.py` |
| `sim_px4.py` | Isaac Sim + PX4 SITL in lockstep: PhysX integrates base and rotor bodies, forces applied where they act; VTOL mission over MAVLink; `--replay` renders a video from a flight log |
| `px4/airframes/22001_isaac_bullet` | PX4 quad-tailsitter airframe; `px4/install_airframe.sh` installs it into PX4-Autopilot |
| `px4/run_mission.sh` | Runs `sim_px4.py`, retrying if PX4 SITL stalls its airspeed selector at arming |
| `sim_bullet.py`, `bullet_control/` | Earlier standalone sim with a Python controller (no PX4) |
| `launch_drone.py` | Imports the source URDF and drops it on a ground plane (gravity only) |
| `tools/` | URDF -> FLU conversion, STL colours -> USD, URDF roll, URDF-vs-CAD check, static margin, trim curves |
| `isaac sim dynamics guide.md` | Simulation guide |

## Setup

```bash
# PX4 SITL (v1.16) and the Bullet airframe
git clone --branch v1.16.0 --recursive https://github.com/PX4/PX4-Autopilot.git ~/PX4-Autopilot
bash ~/PX4-Autopilot/Tools/setup/ubuntu.sh --no-nuttx --no-sim-tools
px4/install_airframe.sh ~/PX4-Autopilot          # copies the airframe, registers it, builds px4_sitl_default

# MAVLink / log libraries for Isaac Sim's Python
/opt/IsaacSim/python.sh -m pip install --target px4/pylib pymavlink pyulog
```

## Run

```bash
px4/run_mission.sh                                                   # GUI: takeoff, transition, dash, land
px4/run_mission.sh --headless --max-thrust --distance 3500 --capture logs/frames   # full-throttle top-speed run + video frames
/opt/IsaacSim/python.sh sim_px4.py --headless --replay logs/px4_flight.csv --capture logs/frames  # re-render a log
/opt/IsaacSim/python.sh tools/plot_trim_curves.py                    # angle of attack and drag vs speed
```

Regenerate the assets after editing the URDF:

```bash
/opt/IsaacSim/python.sh tools/make_flu_urdf.py high-speed-drone.urdf assets/bullet_flu.urdf --mesh-dir meshes
/opt/IsaacSim/python.sh /opt/IsaacSim/standalone_examples/api/isaacsim.asset.importer.urdf/urdf_import.py \
    --urdf assets/bullet_flu.urdf --usd-path assets/usd --no-fix-base --robot-type Aerial \
    --joint-drive-type force --override-joint-stiffness 0 --override-joint-damping 0.01
/opt/IsaacSim/python.sh tools/apply_stl_colors.py assets/usd/bullet_flu assets/bullet_flu.urdf
/opt/IsaacSim/python.sh tools/check_urdf_vs_cad.py assets/bullet_flu.urdf
```

## Status and caveats

- Wing-borne flight in SITL: steady about 71 m/s at full throttle, about 3 deg angle of attack at 50 m/s.
- The CG is 20 mm forward of the spec sheet (URDF rev 2026-10-14e) for static pitch stability.
- Prop spin directions come from the CAD blade pitch (`tools/check_urdf_vs_cad.py`).
- Many aero and propulsion values are placeholders (`hsquad_dynamics/params.py`): mean chord, Cm0,
  aspect ratio, stall, prop C_T/C_Q curves, battery. The CAD wing planform (~0.03 m^2) does not match
  the spec-sheet wing area (0.1094 m^2), and the CAD prop pitch suggests less thrust at high speed than
  the placeholder prop curve, so top speed is likely optimistic.
