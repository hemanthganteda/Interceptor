#!/usr/bin/env python3
"""Guide §5.5: aerodynamic pitching moment about the vehicle CG vs alpha, and dM/dalpha, at
100 / 300 / 500 km/h (plus any speeds given). Propulsion off.

  /opt/IsaacSim/python.sh tools/static_margin.py [--speeds-kmh 100 170 300 500]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hsquad_dynamics import BulletDynamics, BulletParams  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speeds-kmh", type=float, nargs="+", default=[100.0, 300.0, 500.0])
    args = ap.parse_args()
    p = BulletParams()
    dyn = BulletDynamics(p, ref_point=p.cg_total)
    alphas = np.radians(np.arange(-4.0, 12.01, 1.0))
    n = len(alphas)
    print(f"vehicle CG x = {p.cg_total[0]:.4f} m from nose (base_link CG {p.cg_body[0]:.4f}); c_bar = {p.mac} m (PLACEHOLDER)")
    print("pitching moment M (N m, + nose-up) about vehicle CG; dM/da (N m/rad) by central difference")
    print("alpha_deg " + " ".join(f"{a:7.0f}" for a in np.degrees(alphas)))
    for vk in args.speeds_kmh:
        V = vk / 3.6
        vb = V * np.stack([np.cos(alphas), np.zeros(n), -np.sin(alphas)], 1)
        _, M, _ = dyn.aero(vb, np.zeros((n, 3)))
        m_up = -M[:, 1]  # + nose-up (rotation about -y in FLU)
        dm = np.gradient(m_up, alphas)
        print(f"{vk:5.0f} km/h M   " + " ".join(f"{x:7.2f}" for x in m_up))
        print(f"{'':10s} dM/da " + " ".join(f"{x:7.1f}" for x in dm))
        unstable = np.degrees(alphas[dm > 0])
        print(f"{'':10s} {'statically UNSTABLE for alpha in ' + str(unstable.min()) + '..' + str(unstable.max()) + ' deg' if unstable.size else 'statically stable over the range'}")


if __name__ == "__main__":
    main()
