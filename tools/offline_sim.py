#!/usr/bin/env python3
"""Pure-NumPy 6-DOF loop (no Isaac) with the same dynamics + controller + mission.

Used to tune the controller quickly and as a reference for the Isaac run. Ground is a crude
"sitting on the tail" constraint, so only use this for the airborne part.

  python3 tools/offline_sim.py [--speed 60] [--plot out.png]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bullet_control import DashMission, TailsitterController  # noqa: E402
from hsquad_dynamics import BulletDynamics, BulletParams, State  # noqa: E402


def expm_so3(w):
    th = np.linalg.norm(w)
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    if th < 1e-12:
        return np.eye(3) + K
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th**2 * K @ K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=60.0)
    ap.add_argument("--altitude", type=float, default=30.0)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--tmax", type=float, default=None)
    args = ap.parse_args()

    p = BulletParams()
    dyn = BulletDynamics(p, ref_point=p.cg_total)
    ctl = TailsitterController(p)
    m, J, cg = p.mass_total, p.inertia_total, p.cg_total

    R = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])  # nose up, top facing -x
    z_rest = 0.685 + cg[0]  # tail stubs at x = -0.685 from the nose
    pos = np.array([0.0, 0.0, z_rest])
    vel = np.zeros(3)
    w = np.zeros(3)
    s = State.initial(1)
    mission = DashMission(pos, altitude=args.altitude, speed=args.speed)
    dt = args.dt
    n = int((args.tmax or mission.t_end) / dt)
    log = []
    thr = np.zeros(4)
    dg = dict(alpha_d=np.nan, e_R=np.zeros(3), T_cmd=0.0, M_cmd=np.zeros(3))
    on_ground = True
    v_pack = 25.2
    for k in range(n):
        t = k * dt
        s.R_wb[0], s.v_world[0], s.omega_body[0] = R, vel, w
        if k % 4 == 0:  # 250 Hz controller
            ref = mission(t)
            if t < mission.t_climb0:
                thr = np.full(4, 0.06)
                ctl.reset()
            else:
                thr, dg = ctl.compute(4 * dt, pos, vel, R, w, s.rotor_omega[0], v_pack, ref, airborne=not on_ground)
        F_b, M_b, out = dyn.step(s, thr[None], None, dt)
        v_pack = out["v_pack"][0]
        F_w = R @ F_b[0] - m * p.g * np.array([0, 0, 1.0])
        acc = F_w / m
        wdot = (M_b[0] - np.cross(w, J * w)) / J
        vel = vel + acc * dt
        pos = pos + vel * dt
        w = w + wdot * dt
        R = R @ expm_so3(w * dt)
        if pos[2] <= z_rest and vel[2] <= 0:
            on_ground = True
            pos[2] = z_rest
            vel[:] = 0
            w[:] = 0
        else:
            on_ground = False
        if k % 10 == 0:
            V = out["V"][0]
            log.append(
                [t, *pos, *vel, V, np.degrees(out["alpha"][0]), *out["rotor_omega"][0], *out["thrust"][0], v_pack,
                 out["i_bat"][0], s.soc[0], np.degrees(np.arccos(np.clip(R[2, 0], -1, 1))), *(ref["p"]), np.degrees(dg["alpha_d"]), *dg["e_R"], dg["T_cmd"], *dg["M_cmd"]]
            )
        if not np.all(np.isfinite(pos)) or np.linalg.norm(pos) > 1e5:
            print(f"diverged at t={t:.2f}")
            break
    L = np.array(log)
    hdr = "t,x,y,z,vx,vy,vz,V,alpha_deg,w0,w1,w2,w3,T0,T1,T2,T3,v_pack,i_bat,soc,pitch_from_vertical_deg,xr,yr,zr,alpha_d,eRx,eRy,eRz,Tcmd,Mx,My,Mz"
    if args.csv:
        np.savetxt(args.csv, L, delimiter=",", header=hdr, comments="")
    err = np.linalg.norm(L[:, 1:4] - L[:, 21:24], axis=1)
    for ph_t, name in [(mission.t_hover0, "hover"), (mission.t_acc0 + mission.T_acc, "end of dash accel"),
                       (mission.t_dec0, "cruise end"), (mission.t_hover2 + 2, "hover after brake"), (L[-1, 0], "end")]:
        i = min(int(ph_t / (10 * dt)), len(L) - 1)
        r = L[i]
        print(f"{name:20s} t={r[0]:6.2f} pos=({r[1]:7.1f},{r[2]:5.1f},{r[3]:5.1f}) V={r[7]:5.1f} m/s "
              f"alpha={r[8]:6.1f} tilt={r[20]:5.1f} track_err={err[i]:5.2f} m Vpack={r[17]:5.2f} I={r[18]:5.1f}")
    print(f"max track err {err.max():.2f} m, max V {L[:,7].max():.1f} m/s ({L[:,7].max()*3.6:.0f} km/h), "
          f"max |alpha| airborne {np.abs(L[L[:,7]>15,8]).max() if np.any(L[:,7]>15) else 0:.1f} deg, final SOC {L[-1,19]:.3f}")


if __name__ == "__main__":
    main()
