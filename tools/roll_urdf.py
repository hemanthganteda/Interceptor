#!/usr/bin/env python3
"""Roll the airframe geometry about the fuselage (x) axis in the base_link frame.

The CAD meshes were exported with the wing spanning the URDF z axis, so with z up the wing stood
vertical. This rotates everything attached to base_link by R = Rx(deg): visual and collision
origins, the base_link inertia tensor and the origins of the joints whose parent is base_link
(rotors). Frames that define conventions (body_flu) and the CG (on the x axis) are left alone.

  python3 tools/roll_urdf.py high-speed-drone.urdf high-speed-drone.urdf --deg 90
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET

import numpy as np

KEEP_JOINTS = {"body_flu_joint"}


def rpy_to_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def mat_to_rpy(m):
    p = math.asin(max(-1.0, min(1.0, -m[2, 0])))
    if abs(math.cos(p)) > 1e-9:
        return math.atan2(m[2, 1], m[2, 2]), p, math.atan2(m[1, 0], m[0, 0])
    return 0.0, p, math.atan2(-m[0, 1], m[1, 1])


def fmt(v):
    return " ".join(f"{0.0 if abs(x) < 1e-12 else x:.9g}" for x in v)


def rotate_origin(el, R):
    xyz = np.array([float(s) for s in el.get("xyz", "0 0 0").split()])
    rpy = [float(s) for s in el.get("rpy", "0 0 0").split()]
    el.set("xyz", fmt(R @ xyz))
    el.set("rpy", fmt(mat_to_rpy(R @ rpy_to_mat(*rpy))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--deg", type=float, default=90.0)
    ap.add_argument("--link", default="base_link")
    a = ap.parse_args()
    th = math.radians(a.deg)
    R = np.array([[1, 0, 0], [0, math.cos(th), -math.sin(th)], [0, math.sin(th), math.cos(th)]])

    # Text-level edit so comments and layout survive: find the elements with ElementTree, then
    # rewrite only their <origin .../> and <inertia .../> tags in the source text.
    import re
    text = open(a.src).read()
    robot = ET.fromstring(text)
    link = next(l for l in robot.findall("link") if l.get("name") == a.link)
    targets = []  # (element tag, name) whose origin must rotate, in document order
    for child in link:
        if child.tag in ("visual", "collision"):
            targets.append((child.tag, child.get("name")))
    joints = [j.get("name") for j in robot.findall("joint")
              if j.find("parent").get("link") == a.link and j.get("name") not in KEEP_JOINTS]

    def rot_origin_tag(tag):
        el = ET.fromstring(tag if tag.rstrip().endswith("/>") else tag + "</origin>")
        rotate_origin(el, R)
        close = "/>" if tag.rstrip().endswith("/>") else ">"
        return f'<origin xyz="{el.get("xyz")}" rpy="{el.get("rpy")}"{close}'

    def rotate_in_block(txt, open_re):
        m = re.search(open_re, txt)
        assert m, open_re
        o = re.compile(r"<origin\b[^>]*?/?>").search(txt, m.end())
        return txt[:o.start()] + rot_origin_tag(o.group(0)) + txt[o.end():]

    # restrict visual/collision edits to the base_link element
    ls = re.search(r'<link\s+name="%s"\s*>' % re.escape(a.link), text).start()
    le = text.index("</link>", ls)
    block = text[ls:le]
    for tag, name in targets:
        block = rotate_in_block(block, r'<%s\s+name="%s"\s*>' % (tag, re.escape(name)))
    im = re.search(r"<inertia\b[^>]*/>", block, re.S)
    I_el = ET.fromstring(im.group(0))
    k = ["ixx", "ixy", "ixz", "iyy", "iyz", "izz"]
    v = {n: float(I_el.get(n, "0")) for n in k}
    I = np.array([[v["ixx"], v["ixy"], v["ixz"]], [v["ixy"], v["iyy"], v["iyz"]], [v["ixz"], v["iyz"], v["izz"]]])
    I = R @ I @ R.T  # inertial frame orientation unchanged; CG is on the roll axis
    f = lambda x: f"{0.0 if abs(x) < 1e-15 else x:.9e}"
    new_I = (f'<inertia ixx="{f(I[0,0])}" ixy="{f(I[0,1])}" ixz="{f(I[0,2])}"\n'
             f'                                     iyy="{f(I[1,1])}" iyz="{f(I[1,2])}"\n'
             f'                                                           izz="{f(I[2,2])}"/>')
    block = block[:im.start()] + new_I + block[im.end():]
    text = text[:ls] + block + text[le:]
    for jn in joints:
        text = rotate_in_block(text, r'<joint\s+name="%s"[^>]*>' % re.escape(jn))
    open(a.dst, "w").write(text)
    print(f"rolled {a.link} geometry by {a.deg:g} deg about x ({len(targets)} visuals/collisions, "
          f"{len(joints)} joints, inertia) -> {a.dst}")


if __name__ == "__main__":
    main()
