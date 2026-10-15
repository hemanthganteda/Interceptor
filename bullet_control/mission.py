"""Reference trajectory: vertical take-off -> hover -> pitch-over and dash -> brake -> land.

All references are for the vehicle CG in world (Z-up) coordinates.
"""

from __future__ import annotations

import numpy as np


def _s(x):  # cubic smoothstep and its first two derivatives (in normalised time)
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x), 6 * x * (1 - x), 6 - 12 * x


def _s_int(x):  # integral of the cubic smoothstep from 0 to x
    x = np.clip(x, 0.0, 1.0)
    return x**3 - 0.5 * x**4


class DashMission:
    def __init__(
        self,
        start: np.ndarray,
        altitude: float = 30.0,
        speed: float = 60.0,
        accel: float = 6.0,
        decel: float = 5.0,
        cruise_time: float = 6.0,
        heading: float = 0.0,
        t_spinup: float = 1.0,
        climb_time: float = 6.0,
        hover_time: float = 3.0,
        land_speed: float = 1.2,
    ):
        self.p0 = np.asarray(start, dtype=float).copy()
        self.alt = altitude
        self.V = speed
        self.heading = heading
        self.dir = np.array([np.cos(heading), np.sin(heading), 0.0])
        self.t_climb0 = t_spinup
        self.T_climb = climb_time
        self.t_hover0 = self.t_climb0 + climb_time
        self.t_acc0 = self.t_hover0 + hover_time
        self.T_acc = 1.5 * speed / accel  # cubic smoothstep: peak accel = 1.5 V / T
        self.t_cruise0 = self.t_acc0 + self.T_acc
        self.t_dec0 = self.t_cruise0 + cruise_time
        self.T_dec = 1.5 * speed / decel
        self.t_hover2 = self.t_dec0 + self.T_dec
        self.t_land0 = self.t_hover2 + 4.0
        self.T_land = 1.5 * (altitude - self.p0[2]) / land_speed
        self.t_end = self.t_land0 + self.T_land + 3.0
        self.d_acc = speed * self.T_acc * _s_int(1.0)
        self.d_dec = speed * self.T_dec * (1.0 - _s_int(1.0))
        self.d_cruise = speed * cruise_time

    def phase(self, t: float) -> str:
        for name, t0 in (
            ("land", self.t_land0), ("hover", self.t_hover2), ("brake", self.t_dec0), ("cruise", self.t_cruise0),
            ("dash", self.t_acc0), ("hover", self.t_hover0), ("climb", self.t_climb0),
        ):
            if t >= t0:
                return "done" if (name == "land" and t > self.t_land0 + self.T_land + 1.0) else name
        return "spinup"

    def __call__(self, t: float) -> dict:
        p = self.p0.copy()
        v = np.zeros(3)
        a = np.zeros(3)
        # vertical
        if t < self.t_land0:
            x = (t - self.t_climb0) / self.T_climb
            s, ds, dds = _s(x)
            h = self.alt - self.p0[2]
            p[2] = self.p0[2] + h * s
            if 0 < x < 1:
                v[2] = h * ds / self.T_climb
                a[2] = h * dds / self.T_climb**2
        else:
            x = (t - self.t_land0) / self.T_land
            s, ds, dds = _s(x)
            h = self.alt - self.p0[2]
            p[2] = self.alt - (h + 0.15) * s  # aim slightly below the ground so it settles on the stubs
            if 0 < x < 1:
                v[2] = -(h + 0.15) * ds / self.T_land
                a[2] = -(h + 0.15) * dds / self.T_land**2
        # along-track
        if t < self.t_acc0:
            d, sp, ac = 0.0, 0.0, 0.0
        elif t < self.t_cruise0:
            x = (t - self.t_acc0) / self.T_acc
            s, ds, _ = _s(x)
            d, sp, ac = self.V * self.T_acc * _s_int(x), self.V * s, self.V * ds / self.T_acc
        elif t < self.t_dec0:
            d, sp, ac = self.d_acc + self.V * (t - self.t_cruise0), self.V, 0.0
        elif t < self.t_hover2:
            x = (t - self.t_dec0) / self.T_dec
            s, ds, _ = _s(x)
            d = self.d_acc + self.d_cruise + self.V * self.T_dec * (x - _s_int(x))
            sp, ac = self.V * (1 - s), -self.V * ds / self.T_dec
        else:
            d, sp, ac = self.d_acc + self.d_cruise + self.d_dec, 0.0, 0.0
        p += d * self.dir
        v += sp * self.dir
        a += ac * self.dir
        return dict(p=p, v=v, a=a, heading=self.heading)
