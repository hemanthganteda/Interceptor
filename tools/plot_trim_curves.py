#!/usr/bin/env python3
"""Level-flight trim of the Bullet vs airspeed: angle of attack, drag, and the force/velocity geometry.

Trim: flight path level, body pitched up by alpha; thrust along the body x axis balances drag, and
wing+body lift plus the thrust's vertical component balance weight (hsquad_dynamics.aero, current
params). PX4 SITL flights (logs/*.csv) are overlaid where available.

  /opt/IsaacSim/python.sh tools/plot_trim_curves.py            # -> logs/trim_vs_speed.png + .csv
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from hsquad_dynamics import BulletDynamics, BulletParams  # noqa: E402

# palette (reference light instance): entity colours, validated as a set
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # velocity, drag, lift
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"

p = BulletParams()
dyn = BulletDynamics(p, ref_point=p.cg_total)
W = p.mass_total * p.g
alphas = np.radians(np.linspace(-2.0, 16.0, 3601))


def trim(V):
    vb = V * np.stack([np.cos(alphas), np.zeros_like(alphas), -np.sin(alphas)], 1)
    F, _, aux = dyn.aero(vb, np.zeros((len(alphas), 3)))
    ca, sa = np.cos(alphas), np.sin(alphas)
    D = -(F[:, 0] * ca - F[:, 2] * sa)  # along the (horizontal) flight path, rearward
    Lf = F[:, 2] * ca + F[:, 0] * sa  # aero force, vertical
    T = D / ca
    i = np.argmin(np.abs(Lf + T * sa - W))
    a = alphas[i]
    # drag split (same formulas as hsquad_dynamics.aero, pre-stall)
    q = 0.5 * p.rho * V**2
    cl = p.cl_alpha * a
    d_prof = q * p.wing_area * p.cd0_wing
    d_ind = q * p.wing_area * cl**2 / (np.pi * p.oswald * p.aspect_ratio)
    d_body = D[i] - d_prof - d_ind
    return dict(V=V, alpha=np.degrees(a), drag=D[i], thrust=T[i], lift=Lf[i], d_prof=d_prof, d_ind=d_ind, d_body=d_body,
                ok=abs(Lf[i] + T[i] * sa[i] - W) < 0.05 * W and np.degrees(a) < np.degrees(p.alpha_stall))


speeds = np.arange(27.0, 80.01, 0.5)  # below ~27 m/s alpha enters the model's stall blend
rows = [trim(V) for V in speeds]
rows = [r for r in rows if r["ok"]]
S = {k: np.array([r[k] for r in rows]) for k in rows[0] if k != "ok"}

# PX4 SITL flights: steady level wing-borne windows
sim = []
for f, lab in (("logs/px4_50ms.csv", "50 m/s target"), ("logs/px4_cgfwd.csv", "full throttle")):
    path = os.path.join(HERE, f)
    if not os.path.exists(path):
        continue
    L = np.genfromtxt(path, delimiter=",", names=True)
    fw = L["vtol"] == 2
    t0 = L["t"][fw].min() + 10
    m = fw & (L["t"] > t0) & (np.abs(L["vz"]) < 0.5)
    m &= np.abs(L["V"] - np.median(L["V"][m])) < 1.0
    T = sum(L[f"T{k}"][m] for k in range(4))
    sim.append((lab, L["V"][m].mean(), L["alpha_deg"][m].mean(), T.mean()))

# ------------------------------------------------------------------------------------------ figure
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK, "text.color": INK})
fig = plt.figure(figsize=(13, 8.6), facecolor=SURFACE)
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.72], hspace=0.30, wspace=0.18)
axA, axB, axC = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
for ax in (axA, axB):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Airspeed (m/s)")
    ax.set_xlim(speeds[0], speeds[-1])

# A: angle of attack
axA.plot(S["V"], S["alpha"], color=INK, lw=2, solid_capstyle="round")
axA.set_title("Angle of attack for level flight", loc="left", fontweight="bold")
axA.set_ylabel("Angle of attack α (deg)")
axA.axhline(np.degrees(p.alpha_stall), color=MUTED, lw=1, ls=(0, (4, 3)))
axA.text(speeds[-1], np.degrees(p.alpha_stall) + 0.3, "stall (placeholder 15°)", ha="right", color=INK2, fontsize=9)
for lab, V, a, _ in sim:
    axA.plot(V, a, "o", ms=8, color=INK, mfc=SURFACE, mew=2, zorder=5)
    axA.annotate(f"PX4 SITL, {lab}\n{V:.1f} m/s → α {a:.2f}°", (V, a), xytext=(-60, 40), textcoords="offset points",
                 fontsize=9, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
axA.set_ylim(0, 16.5)
axA.text(31.5, 9.3, "model trim (hsquad_dynamics)", color=INK, fontsize=9)

# B: drag
axB.plot(S["V"], S["drag"], color=ORANGE, lw=2.4, solid_capstyle="round", label="Total drag")
comp = [("d_prof", "Wing profile drag (C_D0)", (0, ())), ("d_ind", "Wing induced drag", (0, (5, 3))),
        ("d_body", "Fuselage drag", (0, (1.5, 2.5)))]
for k, lab, ls in comp:
    axB.plot(S["V"], S[k], color=INK2, lw=1.4, ls=ls, label=lab)
axB.set_title("Drag in level flight", loc="left", fontweight="bold")
axB.set_ylabel("Drag (N)")
for lab, V, _, T in sim:
    axB.plot(V, T, "o", ms=8, color=ORANGE, mfc=SURFACE, mew=2, zorder=5, label=None)
    axB.annotate(f"PX4 SITL thrust ≈ drag\n{V:.1f} m/s → {T:.1f} N", (V, T), xytext=(14, -38), textcoords="offset points",
                 fontsize=9, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
axB.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK2)
axB.set_ylim(0, None)

# C: force / velocity geometry at three speeds (angles drawn true to scale)
axC.set_facecolor(SURFACE)
axC.set_aspect("equal")
axC.axis("off")
axC.set_title("Direction of velocity, drag and lift in level flight (angles to scale; arrow lengths schematic)",
              loc="left", fontweight="bold")
picks = [V for V in (30.0, 50.0, 70.0) if V >= S["V"][0]]
for j, Vp in enumerate(picks):
    r = trim(Vp)
    a = np.radians(r["alpha"])
    cx = j * 4.2
    # horizon / flight path
    axC.plot([cx - 1.7, cx + 1.7], [0, 0], color=GRID, lw=1, zorder=1)
    # fuselage axis, nose to the right, pitched up by alpha
    u = np.array([np.cos(a), np.sin(a)])
    axC.plot([cx - 1.2 * u[0], cx + 1.2 * u[0]], [-1.2 * u[1], 1.2 * u[1]], color=INK, lw=5, solid_capstyle="round", zorder=3)
    axC.text(cx + 1.22 * u[0], 1.22 * u[1] + 0.16, "nose", fontsize=8.5, color=INK2, ha="center", va="bottom")
    # velocity: along the flight path (horizontal, forward)
    axC.annotate("", (cx + 1.55, 0), (cx, 0), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.2, mutation_scale=16), zorder=4)
    axC.text(cx + 1.55, -0.40, "velocity V\n(flight path, 0°)", color=INK2, fontsize=8.5, ha="right", va="top")
    # drag: opposite to velocity, i.e. horizontal, rearward
    axC.annotate("", (cx - 1.55, 0), (cx, 0), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.2, mutation_scale=16), zorder=4)
    axC.text(cx - 1.55, -0.40, f"drag {r['drag']:.1f} N\n(along −V, 180°)", color=INK2, fontsize=8.5, ha="left", va="top")
    # lift: perpendicular to velocity
    axC.annotate("", (cx, 1.3), (cx, 0), arrowprops=dict(arrowstyle="-|>", color=AQUA, lw=2.2, mutation_scale=16), zorder=4)
    axC.text(cx + 0.08, 1.33, f"lift {r['lift']:.1f} N (⊥ V)", color=INK2, fontsize=8.5, ha="left", va="bottom")
    # alpha arc between flight path and fuselage axis (exaggerated radius so it is visible)
    th = np.linspace(0, a, 30)
    axC.plot(cx + 0.95 * np.cos(th), 0.95 * np.sin(th), color=INK, lw=1.2, zorder=4)
    mid = a / 2
    axC.annotate(f"α = {r['alpha']:.1f}°", (cx + 0.95 * np.cos(mid), 0.95 * np.sin(mid)), xytext=(cx + 0.35, 0.62),
                 fontsize=9.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    axC.text(cx, -1.08, f"{Vp:.0f} m/s ({Vp * 3.6:.0f} km/h)", ha="center", fontsize=10.5, fontweight="bold", color=INK)
    axC.text(cx, -1.34, f"drag acts {r['alpha']:.1f}° below the fuselage axis (rearward)",
             ha="center", fontsize=8.5, color=INK2)
axC.set_xlim(-2.0, (len(picks) - 1) * 4.2 + 2.0)
axC.set_ylim(-1.45, 1.7)

fig.text(0.01, 0.005, f"Bullet {p.mass_total:.2f} kg, wing {p.wing_area} m² (spec), C_Lα {p.cl_alpha}/rad, CG x {p.cg_total[0]:.4f} m. "
         "Placeholder aero (c̄, AR, e, stall) per hsquad_dynamics/params.py. Thrust along the fuselage balances drag; "
         "its small vertical part (T·sin α) is included in the lift balance.", fontsize=8, color=MUTED)
out = os.path.join(HERE, "logs/trim_vs_speed.png")
fig.savefig(out, dpi=110, facecolor=SURFACE, bbox_inches="tight")
np.savetxt(out.replace(".png", ".csv"), np.column_stack([S[k] for k in ("V", "alpha", "drag", "d_prof", "d_ind", "d_body", "lift", "thrust")]),
           delimiter=",", fmt="%.4f", header="airspeed_m_s,alpha_deg,drag_N,wing_profile_N,wing_induced_N,body_N,aero_lift_N,thrust_N",
           comments="")
print(f"wrote {out} and .csv; {len(rows)} trim points {S['V'][0]:.1f}..{S['V'][-1]:.1f} m/s; sim points: "
      + "; ".join(f"{l} V={V:.1f} a={a:.2f} T={T:.1f}" for l, V, a, T in sim))
