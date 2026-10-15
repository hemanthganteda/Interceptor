#!/usr/bin/env python3
"""Check a URDF against its CAD meshes: place every visual mesh with its URDF transform and compare
with rotor joints and collision primitives; infer each propeller's spin sense from its blade pitch.

  /opt/IsaacSim/python.sh tools/check_urdf_vs_cad.py assets/bullet_flu.urdf     # FLU, thrust along +x
"""
import math
import os
import struct
import sys
import xml.etree.ElementTree as ET

import numpy as np


def rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])


def T_of(el):
    T = np.eye(4)
    if el is not None:
        T[:3, :3] = rpy(*[float(s) for s in el.get("rpy", "0 0 0").split()])
        T[:3, 3] = [float(s) for s in el.get("xyz", "0 0 0").split()]
    return T


def stl(path):
    b = open(path, "rb").read()
    n = struct.unpack("<I", b[80:84])[0]
    d = np.frombuffer(b[84:84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return d["v"].reshape(-1, 3).astype(float), d["a"].repeat(3)


def apply(T, P):
    return P @ T[:3, :3].T + T[:3, 3]


urdf = sys.argv[1]
base_dir = os.path.dirname(os.path.abspath(urdf))
root = ET.parse(urdf).getroot()
links = {l.get("name"): l for l in root.findall("link")}
joint_T = {j.find("child").get("link"): T_of(j.find("origin")) for j in root.findall("joint")}
axis = {j.find("child").get("link"): np.array([float(s) for s in j.find("axis").get("xyz").split()])
        for j in root.findall("joint") if j.find("axis") is not None}


def meshes(link_name):
    out = {}
    for v in links[link_name].findall("visual"):
        m = v.find("geometry/mesh")
        if m is None:
            continue
        fn = m.get("filename")
        fn = fn if os.path.isabs(fn) else os.path.join(base_dir, fn)
        s = float(m.get("scale", "1 1 1").split()[0])
        P, A = stl(fn)
        out[v.get("name")] = (apply(T_of(v.find("origin")), P * s), A)
    return out


base = meshes("base_link")
air, acol = base["airframe_visual"]
print("== airframe (base_link frame)")
for code, name in ((0x1BC, "orange wing"), (0x7FE0, "cyan booms/pylons")):
    P = air[acol == code]
    ext = P.max(0) - P.min(0)
    print(f"   {name:18s} extent x {ext[0]*1000:5.0f}  y {ext[1]*1000:5.0f}  z {ext[2]*1000:5.0f} mm"
          f"  -> spans {'y (horizontal)' if ext[1] > ext[2] else 'z (vertical)'}")

print("== rotors: joint origin vs motor mesh and prop hub")
mot = {k: v[0] for k, v in base.items() if k.startswith("motor")}
for i in range(4):
    ln = f"rotor_{i}"
    J = joint_T[ln]
    hub = J[:3, 3]
    prop = apply(J, meshes(ln)[f"rotor_{i}_visual"][0])
    # nearest motor mesh centroid in y-z
    best = min(mot.items(), key=lambda kv: np.linalg.norm(kv[1][:, 1:].mean(0) - hub[1:]))
    mc = best[1].mean(0)
    pc = prop.mean(0)
    ax = J[:3, :3] @ axis[ln]
    print(f"   {ln}: joint ({hub[0]:+.4f},{hub[1]:+.4f},{hub[2]:+.4f})  {best[0]} centre offset y/z "
          f"{(mc[1]-hub[1])*1000:+.1f}/{(mc[2]-hub[2])*1000:+.1f} mm  prop centroid offset "
          f"{np.linalg.norm(pc[1:]-hub[1:])*1000:.1f} mm radial, {(pc[0]-hub[0])*1000:+.1f} mm axial; joint axis {np.round(ax, 3)}")

print("== collision boxes/cylinders vs airframe mesh (fraction of primitive centre region inside mesh bbox)")
for c in links["base_link"].findall("collision"):
    T = T_of(c.find("origin"))
    g = c.find("geometry")[0]
    if g.tag == "box":
        sz = np.array([float(s) for s in g.get("size").split()])
    else:
        r, L = float(g.get("radius")), float(g.get("length"))
        sz = np.array([2 * r, 2 * r, L])
    corners = np.array([[sx, sy, sz_] for sx in (-.5, .5) for sy in (-.5, .5) for sz_ in (-.5, .5)]) * sz
    W = apply(T, corners)
    lo, hi = W.min(0), W.max(0)
    inside = air[np.all((air >= lo) & (air <= hi), axis=1)]
    print(f"   {c.get('name'):14s} world bbox x {lo[0]*1000:5.0f}..{hi[0]*1000:5.0f} y {lo[1]*1000:5.0f}..{hi[1]*1000:5.0f} "
          f"z {lo[2]*1000:5.0f}..{hi[2]*1000:5.0f}  mesh vertices inside: {len(inside)}")

print("== propeller spin sense from blade pitch (thrust along +x of this file; rotor spin axis = link x)")
for i in range(4):
    v = links[f"rotor_{i}"].find("visual")
    m_el = v.find("geometry/mesh")
    fn = m_el.get("filename")
    fn = fn if os.path.isabs(fn) else os.path.join(base_dir, fn)
    P, _ = stl(fn)
    Q = apply(T_of(v.find("origin")), P * float(m_el.get("scale", "1 1 1").split()[0]))  # rotor link frame
    R = np.hypot(Q[:, 1], Q[:, 2]).max()
    span_axis = 2 if np.ptp(Q[:, 2]) > np.ptp(Q[:, 1]) else 1  # blades lie along link z or y
    chord_axis = 3 - span_axis
    votes, pitches = [], []
    for sgn in (+1, -1):  # the two blades
        for f0 in (0.45, 0.6, 0.75):
            m = (sgn * Q[:, span_axis] > f0 * R) & (sgn * Q[:, span_axis] < (f0 + 0.09) * R)
            slope = np.polyfit(Q[m, chord_axis], Q[m, 0], 1)[0]  # axial vs chordwise along the section
            e_span = np.zeros(3); e_span[span_axis] = sgn
            t = np.cross([1.0, 0.0, 0.0], e_span)[chord_axis]   # motion direction (chord axis) for spin +1
            # leading edge is upstream (+x) and ahead in the direction of motion
            votes.append(int(np.sign(slope * t)))
            pitches.append(np.degrees(np.arctan(abs(slope))))
    s_ = int(np.sign(sum(votes)))
    print(f"   rotor_{i}: spin {s_:+d} about +x (blade-section votes {votes}, pitch {np.mean(pitches):.1f} deg)")
