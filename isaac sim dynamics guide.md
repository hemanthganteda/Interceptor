# Simulating the Bullet Airframe in Isaac Sim

*Draft v0.1, 14 October 2026. Targets Isaac Sim 5.x, Isaac Lab 2.x, Pegasus Simulator 5.1.x, PX4 1.15/1.16. Uses `bullet_v2.urdf` and the CFD spec sheet (3.3626 kg configuration).*

---

## 1. What Isaac Sim does and does not give you

Isaac Sim is a PhysX rigid-body simulator with a Python API. It will integrate the 6-DOF equations of motion, handle the rotor joints, and provide contact and collision. It does **not** ship a lift-drag or motor plugin like Gazebo's. Every aerodynamic and propulsive force on this vehicle must be computed in your own Python code each physics step and applied to the rigid body. That is more work than Gazebo up front, but it is also why Isaac Sim fits this project: the force model is yours, it runs vectorised across thousands of environments in Isaac Lab, and the same code is reused for PX4-in-the-loop through Pegasus.

The architecture is therefore:

```
USD stage (from URDF)             <- geometry, mass, inertia, rotor joints
        |
PhysX rigid body (base_link)      <- integrates F and M you apply
        |
hsquad_dynamics (Python/Torch)    <- aero tables, rotor model, motor, ESC, battery
        |
   +----+--------------------+
   |                         |
Pegasus + PX4 SITL      Isaac Lab RL env
(validation, 1 vehicle) (training, 4096 envs)
```

---

## 2. Frame conventions (read before importing anything)

Three frames are in play.

| Frame | x | y | z | Used by |
|---|---|---|---|---|
| URDF (CAD) | aft | starboard | up | `bullet_v2.urdf`, meshes |
| FLU | forward | port | up | PX4, Pegasus, Isaac Lab quadrotor examples |
| World (Isaac) | — | — | up (Z-up stage) | PhysX, USD |

Decision: **convert the URDF to FLU before import.** Pegasus's `Multirotor` class and every Isaac Lab drone example assume body x forward, and fighting that in every force function invites sign errors. The conversion is a 180° yaw about z: `x_flu = −x_urdf`, `y_flu = −y_urdf`, `z_flu = z_urdf`.

In FLU, with origin kept at the nose apex:

| Item | URDF (aft +x) | FLU (forward +x) |
|---|---|---|
| CG | (0.3071, 0, 0) | (−0.3071, 0, 0) |
| Wing c/4 | (0.2776, 0, 0) | (−0.2776, 0, 0) |
| Body CP, α = 0° | (0.392, 0, 0) | (−0.392, 0, 0) |
| Body CP, α = 2° | (0.329, 0, 0) | (−0.329, 0, 0) |
| Motor hub end | x = 0.500 | x = −0.500 |
| Rotor joints | (0.5482, ±0.1377, ±0.1377) | (−0.5482, ∓0.1377, ±0.1377) |
| Inertia | Ixx 0.0496, Iyy 0.1031, Izz 0.1037 | unchanged (180° yaw leaves a diagonal tensor alone; flips sign of Ixz and Iyz if they are ever non-zero) |

Note on rotor placement: the rotors sit at ±0.1377 in **both** y and z, i.e. this is an X-configuration viewed from the front, with the thrust axis along the fuselage. In FLU the rotor axes are `(1, 0, 0)` — thrust pushes the vehicle forward along its own x. That is the "missile with four props on the tail" layout, and it means the usual quadrotor assumption (thrust along body z) does not apply. Every control-allocation and force function must use the actual rotor axes, not an inherited quadrotor template.

Simplest way to do the conversion: a `<link name="base_link_flu"/>` already exists in spirit as `body_flu`; make it the root instead. Practically, generate the FLU URDF with a short script that negates x and y of every `<origin xyz>` and every `<axis xyz>`, and sets the root. Keep both files; the URDF importer only sees the FLU one.

---

## 3. Importing the model

### 3.1 Prepare the URDF
1. Mesh paths: Isaac's importer does not resolve `package://`. Replace with absolute or relative paths, or set the importer's package root.
2. Collisions: the primitive boxes and cylinders are fine for PhysX. Do not use the 135k-triangle collision mesh.
3. Rotor joints: keep them `continuous`. The importer turns them into revolute joints with no limits; you will drive them for visual spin only and apply thrust as an external force (§5.2).
4. Strip the `<gazebo>` block. Isaac ignores it; the values move into `hsquad_dynamics`.

### 3.2 Import
Isaac Sim → File → Import → URDF, or in Python:

```python
from isaacsim.asset.importer.urdf import _urdf
cfg = _urdf.ImportConfig()
cfg.merge_fixed_joints = True        # body_flu and any fixed frames collapse into base_link
cfg.fix_base = False
cfg.import_inertia_tensor = True     # use our tensor, not one computed from collision shapes
cfg.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_VELOCITY
cfg.default_drive_strength = 0.0
cfg.default_position_drive_damping = 0.0
cfg.self_collision = False
cfg.make_default_prim = True
cfg.create_physics_scene = True
cfg.distance_scale = 1.0
```

`import_inertia_tensor = True` is the critical flag. Without it PhysX recomputes mass properties from the collision geometry and discards the spec-sheet inertia.

### 3.3 Verify the import
Before anything else, read back what PhysX thinks the body is:

```python
from pxr import UsdPhysics, PhysxSchema
prim = stage.GetPrimAtPath("/World/bullet/base_link")
mass_api = UsdPhysics.MassAPI(prim)
print(mass_api.GetMassAttr().Get())                # expect 3.3626
print(mass_api.GetCenterOfMassAttr().Get())        # expect (-0.3071, 0, 0) in FLU link frame
print(mass_api.GetDiagonalInertiaAttr().Get())     # expect (0.0496, 0.1031, 0.1037)
print(mass_api.GetPrincipalAxesAttr().Get())
```

If the diagonal inertia does not match, the importer applied principal-axis rotation; check `GetPrincipalAxesAttr` is identity. If the CG is wrong, the `merge_fixed_joints` step shifted the link frame; re-check the root link choice.

Drop test: release the vehicle from 2 m with no forces applied, log the fall. Expect 9.81 m/s² and no rotation. Any rotation means the CG and the body frame origin disagree somewhere.

---

## 4. Physics settings

| Setting | Value | Reason |
|---|---|---|
| Physics dt | 1/1000 s | Rate loop at 1 kHz, motor time constant ~10 ms |
| Rendering dt | 1/60 s (or off for headless RL) | |
| Solver | TGS | Better joint stability than PGS |
| Position/velocity iterations | 8 / 2 | Rotor joints are light; default is enough |
| Gravity | (0, 0, −9.81) | Z-up stage |
| Max linear velocity on base_link | ≥ 300 m/s | PhysX default clamps can be lower; set `physxRigidBody:maxLinearVelocity` |
| Max angular velocity | ≥ 100 rad/s | Same reason |
| Sleep threshold | 0 | A vehicle in free flight must never be put to sleep |
| CCD | On for base_link | 190 m/s through a 1 mm ground plane needs it |
| GPU dynamics | On for Isaac Lab; off for Pegasus single-vehicle | |

Substepping: if the motor/ESC electrical model needs 5–10 kHz, run it inside your force function with its own sub-loop; do not raise the PhysX rate, since the rigid body does not need it.

---

## 5. The force model

All of this lives in one package, `hsquad_dynamics`, with a single entry point:

```python
F_body, M_body, rotor_omega = model.step(state, motor_cmd, wind, dt)
```

`state` carries position, orientation, body velocity, body rates, battery SOC and motor RPMs. `F_body` and `M_body` are in FLU body axes, about the CG. Written in PyTorch so one implementation serves both Pegasus (batch size 1) and Isaac Lab (batch size N).

### 5.1 Airspeed and angles
```
v_air_body = R_wb^T (v_world − v_wind)
V   = |v_air_body|
α   = atan2(−v_air_body.z, v_air_body.x)        # positive nose-up in FLU
β   = asin(v_air_body.y / V)
q   = 0.5 ρ V²
```

### 5.2 Rotors
For each of the four rotors, axis `a_i = (1, 0, 0)` in FLU (verify against the imported joint axes), position `r_i` relative to the CG.

Inflow: `V_axial = v_air_body · a_i`; advance ratio `J = V_axial / (n_i d)`.

Thrust and torque from tables `C_T(J, M_tip)`, `C_Q(J, M_tip)`:
```
T_i = C_T ρ n_i² d⁴
Q_i = C_Q ρ n_i² d⁵ · (±1 for handedness)
F_i = T_i a_i
M_i = r_i × F_i  +  Q_i a_i  +  H_i (in-plane force, if the table has it)
```

Until the BEMT tables exist, use the linear fallback `C_T = C_T0 (1 − J/J₀)` from the sizing document, clamped at zero. Do not use a constant `C_T`; it overstates thrust at speed by a factor of two or more.

Apply with `apply_forces_and_torques_at_pos` on the base_link rigid prim (Isaac Lab `RigidObject`) with `is_global=False`. Set the rotor joint velocity targets to `±n_i · 2π` for the visual, nothing more.

### 5.3 Wing
From the spec sheet, wing area `S_w = 0.1094 m²`, c/4 at `r_w = (−0.2776 + 0.3071, 0, 0) = (0.0295, 0, 0)` from the CG — the wing's quarter chord is **29.5 mm ahead of the CG**.

```
CL = 3.621 α           (clamped at α_stall; replace with CL(α) table)
CD = 0.0528 + CL²/(π e AR)      (AR and e from the engineer; placeholder e = 0.85)
Cm = Cm0 + (−0.6) α             (Cm0 from the engineer; assumed 0)
L  = q S_w CL,  D = q S_w CD,  M_w = q S_w c̄ Cm     (c̄ = mean aerodynamic chord, needed)
```
Lift acts perpendicular to the airflow in the x–z plane, drag along it. Transform both into body axes, apply at `r_w`, add `M_w` about y.

Because the wing's c/4 is ahead of the CG, its lift contributes a nose-up (destabilising) moment slope of `3.621 × 0.0295 / c̄` per radian, partly offset by the −0.6 Cmα. Whether the sum is stable depends on c̄ — ask for it.

### 5.4 Fuselage
Frontal area `S_b = 0.01457 m²`, `CD0 = 0.00703` on that area (this number is very low for a body with exposed motors; confirm it is not on wing area).

Normal force: the sheet gives no slope. Use slender-body `CNα ≈ 2.0 /rad` on `S_b` until CFD supplies it. Apply at the CP: `x_cp(α)` from the table (−0.392 at 0°, −0.329 at 2° in FLU-from-nose; i.e. 85 mm and 22 mm **behind** the CG). Interpolate linearly; the 63 mm shift over 2° is steep and a 4° point is needed.

Sideslip: mirror the same model in the x–y plane with `CYβ = −CNα` until β data exist.

### 5.5 Combined static margin check
The spec sheet implies that at α = 0° the body's CP is behind the CG (stable) and the wing's c/4 is ahead of it (destabilising). Compute `dM/dα` at 100, 300, 500 km/h from the model and plot it before any controller work. If it crosses zero inside the envelope, the controller sees a plant that changes sign.

### 5.6 Motor, ESC, battery
```
V_eff  = throttle · V_pack(SOC, I)
I_i    = (V_eff − n_i / KV) / R          (per motor, Y-equivalent)
τ_e    = K_t I_i − Q_i − b n_i
dn_i/dt = τ_e / J_rotor
```
With `R = 40.6 mΩ`, `KV = 700`, `J_rotor` from the URDF rotor inertia (4.75e-5 kg·m² about the spin axis — note the URDF lists it as `ixx`, which is correct since the rotor spins about x). Battery as Thevenin: `V_pack = OCV(SOC) − I R0 − V_rc`, 6S from the CAD rev (Tattu 5100 6S) or whatever the final pack is.

Run this at 5 kHz inside the step function (5 sub-steps per physics step).

### 5.7 Sensors
Isaac Sim provides IMU and camera prims. For the IMU, place the sensor prim at the FC position and let Isaac compute the lever-arm terms, or compute `a_meas = a_cg + ω̇ × r + ω × (ω × r)` yourself and add noise. GPS: body position plus latency buffer and noise. Pitot: `q` with a first-order lag. Feed these to PX4 through Pegasus's sensor interface.

---

## 6. Pegasus Simulator for PX4-in-the-loop

Pegasus wraps a vehicle prim with a `Multirotor` class that applies a stock thrust/drag model and bridges sensors and actuator outputs to PX4 via MAVLink. Replace its physics, keep its plumbing.

1. Install Pegasus (Isaac Sim 5.1 branch), PX4-Autopilot, and run one of the example scripts to confirm the MAVLink link and lockstep work.
2. Create `BulletVehicle(Vehicle)` subclass:
   - Load your USD instead of the Iris.
   - Override `update(dt)`: read the four normalised PX4 actuator outputs, call `hsquad_dynamics.step`, apply forces. Do not instantiate Pegasus's `QuadraticThrustCurve` or `LinearDrag`.
   - Keep Pegasus's `IMU`, `GPS`, `Barometer`, `Magnetometer` sensor classes; add a `Pitot` class that publishes `HIL_SENSOR.diff_pressure`.
3. PX4 side:
   - New airframe with `CA_ROTOR0..3_PX/PY/PZ` = rotor positions relative to CG in FRD (`x = 0.5482 − 0.3071 = 0.241` **aft**, so `PX = −0.241`; `PY = ±0.1377`; `PZ = ∓0.1377`) and `CA_ROTORn_AX/AY/AZ = (1, 0, 0)` — thrust along body x. PX4's control allocator supports arbitrary rotor axes; the vehicle type is still "multicopter".
   - `CA_ROTORn_CT`, `CA_ROTORn_KM` from the static thrust-stand numbers.
   - `COM_OF_LOSS_T` raised; `SYS_HITL` off; lockstep on.
   - `EKF2_IMU_POS_X/Y/Z` = FC offset from CG.
4. Validate: hover (the vehicle hangs nose-up on its props), then a pitch-over to level flight at 50 m/s, then the acceleration profile. Compare against the same run in Gazebo with `bullet_v2.urdf` — the two should agree to within the differences in their aero plugins, which is itself a useful check on the Gazebo LiftDrag placeholders.

---

## 7. Isaac Lab for RL

1. Create a `DirectRLEnv` with `RigidObject` for the vehicle (no articulation needed if the rotors are visual only; use `Articulation` if you want the joints to spin).
2. `_apply_action`: convert the policy output to motor commands (residual on top of the baseline controller's RPM setpoints), call `hsquad_dynamics.step` batched over `num_envs`, then `apply_forces_and_torques_at_pos`.
3. Observations: `v_air_body`, quaternion, body rates, four RPMs, `V_pack`, SOC, distance to gate, last 0.2 s history.
4. Domain randomisation per environment at reset: mass ±5 %, CG x ±10 mm, CG z ±5 mm, inertia ±15 %, every aero coefficient ±20 %, `R_phase` ±20 %, `R0` battery ±30 %, wind.
5. Rewards and termination per the program plan §3c.
6. Export the trained policy to ONNX; validate it in Pegasus with PX4 in the loop before any flight.

CG randomisation is done by re-setting the `RigidObject` mass properties (`set_coms`, `set_masses`, `set_inertias`) at reset, not by moving geometry.

---

## 8. Validation ladder

| Step | Test | Pass criterion |
|---|---|---|
| 1 | Import readback (§3.3) | mass, CG, inertia match spec to 0.1 % |
| 2 | Drop test, no forces | 9.81 m/s², zero rotation |
| 3 | Single-rotor spin-up, vehicle fixed | RPM step response matches bench time constant |
| 4 | Static thrust, all four, vehicle fixed | Total thrust within 5 % of stand |
| 5 | Free release with trim thrust | Hangs nose-up, drifts < 0.1 rad in 5 s |
| 6 | Constant-velocity tow at 100, 300, 500 km/h, no control | Log net pitching moment; sign and magnitude match the hand calculation of §5.5 |
| 7 | Same profile in Gazebo | Agreement to within aero-plugin differences |
| 8 | PX4 hover and pitch-over in Pegasus | Stable with baseline gains |
| 9 | Isaac Lab, 4096 envs, random policy | No NaNs, throughput > 50k steps/s on one GPU |

Steps 1 to 6 need no PX4 and no RL and should be done first; they are where every frame and sign error shows up.

---

## 9. Data still required from the CFD engineer

1. Confirmation that the spec-sheet inertias are about the nose, and a CG-referenced full tensor with products of inertia.
2. Whether 3.3626 kg includes the four propellers.
3. Mean aerodynamic chord, aspect ratio and span efficiency of the wing; Cm0.
4. Wing CL, CD, Cm vs α through stall.
5. Body CNα and a CP-vs-α table to at least 6°; CD0 reference area confirmation.
6. Sideslip derivatives (CYβ, Cnβ, Clβ) for body and wing.
7. Mach dependence of all of the above between M 0.1 and 0.45.
8. Static thrust-stand data for the actual prop: C_T, C_Q at three RPMs; motor time constant.

Items 1–3 block the import validation; 4–8 can be filled with placeholders while the pipeline is built.
