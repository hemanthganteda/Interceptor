"""Baseline flight controller for the Bullet tail-sitter (thrust along body +x, FLU).

Cascade:
  1. Position/velocity PID           -> desired world force F_des = m (a_cmd + g z)
  2. Aero-aware attitude solve       -> body x (thrust axis) chosen by a 1-D search over angle of
                                        attack so that thrust*b1 + F_aero(alpha) = F_des
  3. Geometric SO(3) attitude loop   -> body moment (with model aero moment feed-forward)
  4. Allocation on the real rotor axes and positions -> per-rotor thrust
  5. Thrust -> rotor speed (inverting C_T(J)) -> throttle (inverting the motor model)

Single vehicle, NumPy. The plant model used for feed-forward is hsquad_dynamics with a moment
reference at the full-vehicle CG.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hsquad_dynamics import BulletDynamics, BulletParams

Z = np.array([0.0, 0.0, 1.0])


def vee(m: np.ndarray) -> np.ndarray:
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


def smoothstep01(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def _unit(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else fallback


@dataclass
class Gains:
    kp: np.ndarray = None
    kv: np.ndarray = None
    ki: np.ndarray = None
    i_limit: float = 3.0  # m/s^2 of integral authority
    a_max_xy: float = 12.0
    a_max_z: float = 8.0
    att_wn: np.ndarray = None  # rad/s, attitude natural frequency per axis
    att_zeta: float = 0.8
    omega_p: float = 3.0  # rotor speed P gain (V per V of speed error)

    def __post_init__(self):
        self.kp = np.array([1.2, 1.2, 1.8]) if self.kp is None else self.kp
        self.kv = np.array([1.8, 1.8, 2.6]) if self.kv is None else self.kv
        self.ki = np.array([0.15, 0.15, 0.4]) if self.ki is None else self.ki
        self.att_wn = np.array([14.0, 12.0, 12.0]) if self.att_wn is None else self.att_wn


class TailsitterController:
    def __init__(self, params: BulletParams | None = None, gains: Gains | None = None):
        self.p = params or BulletParams()
        self.g = gains or Gains()
        self.cg = self.p.cg_total
        self.m = self.p.mass_total
        self.J = self.p.inertia_total
        self.model = BulletDynamics(self.p, ref_point=self.cg)
        self.alpha_grid = np.radians(np.arange(-30.0, 170.0, 0.5))
        self.reset()

    def reset(self):
        self.e_int = np.zeros(3)
        self.R_d_prev = None
        self.w_d = np.zeros(3)
        self.alpha_prev = 0.0
        self.u_prev = None

    # ---------------------------------------------------------------- attitude solve
    def _aero_world_grid(self, v_air: np.ndarray, u: np.ndarray, vhat: np.ndarray, V: float):
        a = self.alpha_grid
        n = a.shape[0]
        v_body = V * np.stack([np.cos(a), np.zeros(n), -np.sin(a)], 1)
        F_b, _, _ = self.model.aero(v_body, np.zeros((n, 3)))
        b1 = np.cos(a)[:, None] * vhat + np.sin(a)[:, None] * u
        b3 = -np.sin(a)[:, None] * vhat + np.cos(a)[:, None] * u
        b2 = np.cross(b3, b1)
        F_w = F_b[:, 0:1] * b1 + F_b[:, 1:2] * b2 + F_b[:, 2:3] * b3
        return F_w, b1

    def solve_thrust_axis(self, F_des: np.ndarray, v_air: np.ndarray):
        """Return the desired thrust axis b1 (world), the predicted angle of attack and the span axis
        b2 (= u x vhat) that keeps the wing's lift in the plane of (airspeed, F_des)."""
        V = np.linalg.norm(v_air)
        F_hat = _unit(F_des, Z)
        if V < 1.0:
            return F_hat, np.nan, None
        vhat = v_air / V
        # "lift side" reference u (unit, perpendicular to the airspeed). It follows the force command
        # but keeps continuity when F_des is nearly parallel to the airspeed (no roll flips); a
        # downward push is then met with negative alpha, not by rolling inverted.
        u_ref = self.u_prev if self.u_prev is not None else Z
        u_ref = _unit(u_ref - u_ref.dot(vhat) * vhat, _unit(Z - Z.dot(vhat) * vhat, np.array([1.0, 0, 0])))
        F_perp = F_hat - F_hat.dot(vhat) * vhat
        if F_perp.dot(u_ref) < 0:
            F_perp = F_perp - 2 * F_perp.dot(u_ref) * u_ref  # reflect onto the current lift side
        u = _unit(F_perp + 0.3 * u_ref, u_ref)
        self.u_prev = u
        F_a, b1 = self._aero_world_grid(v_air, u, vhat, V)
        R_res = F_des[None] - F_a
        T = np.einsum("ij,ij->i", R_res, b1)
        miss = np.linalg.norm(R_res - T[:, None] * b1, axis=1) + 10.0 * np.clip(-T, 0, None)
        # prefer continuity with the last solution when the cost is nearly flat (multiple roots)
        miss = miss + 0.5 * np.abs(self.alpha_grid - self.alpha_prev)
        k = int(np.argmin(miss))
        alpha = self.alpha_grid[k]
        if 0 < k < len(miss) - 1:  # parabolic refinement
            c0, c1, c2 = miss[k - 1], miss[k], miss[k + 1]
            den = c0 - 2 * c1 + c2
            if den > 1e-9:
                alpha += 0.5 * (c0 - c2) / den * (self.alpha_grid[1] - self.alpha_grid[0])
        self.alpha_prev = alpha
        return np.cos(alpha) * vhat + np.sin(alpha) * u, alpha, np.cross(u, vhat)

    # ---------------------------------------------------------------- main
    def compute(self, dt, pos, vel, R, omega, rotor_omega, v_pack, ref, wind=None, airborne=True):
        """pos/vel of the full-vehicle CG (world), R body->world, omega body rates.
        ref: dict with p, v, a (world), heading (rad).
        Returns throttle (4,), diag dict.
        """
        g = self.g
        wind = np.zeros(3) if wind is None else wind
        e_p = ref["p"] - pos
        e_v = ref["v"] - vel
        if airborne:
            self.e_int = np.clip(self.e_int + e_p * dt, -g.i_limit / np.maximum(g.ki, 1e-6), g.i_limit / np.maximum(g.ki, 1e-6))
        a_cmd = ref["a"] + g.kp * e_p + g.kv * e_v + g.ki * self.e_int
        a_xy = a_cmd[:2]
        n_xy = np.linalg.norm(a_xy)
        if n_xy > g.a_max_xy:
            a_cmd[:2] *= g.a_max_xy / n_xy
        a_cmd[2] = np.clip(a_cmd[2], -g.a_max_z, g.a_max_z)
        F_des = self.m * (a_cmd + self.p.g * Z)

        # desired attitude
        v_air = vel - wind
        b1_d, alpha_d, b2_plane = self.solve_thrust_axis(F_des, v_air)
        V = np.linalg.norm(v_air)
        psi = ref.get("heading", 0.0)
        left = np.array([-np.sin(psi), np.cos(psi), 0.0])
        # hover: span axis from the commanded heading; wing-borne: span axis from the flight plane
        w = 0.0 if b2_plane is None else smoothstep01((V - 6.0) / 6.0)
        b2_ref = left if w == 0.0 else (1 - w) * left + w * _unit(b2_plane, left)
        b2_d = _unit(b2_ref - b2_ref.dot(b1_d) * b1_d, _unit(np.cross(Z, b1_d), np.array([0, 1.0, 0])))
        b3_d = np.cross(b1_d, b2_d)
        R_d = np.stack([b1_d, b2_d, b3_d], 1)

        if self.R_d_prev is not None:
            w_raw = vee(R_d.T @ (R_d - self.R_d_prev)) / dt
            self.w_d += (1 - np.exp(-dt * 25.0)) * (np.clip(w_raw, -6, 6) - self.w_d)
        self.R_d_prev = R_d

        # feed-forward of the current aero wrench (about the vehicle CG, body axes)
        v_air_b = R.T @ v_air
        F_a, M_a, aux = self.model.aero(v_air_b[None], omega[None])
        F_a, M_a = F_a[0], M_a[0]

        # collective thrust along the actual body x
        F_need_b = R.T @ F_des - F_a
        T_cmd = max(F_need_b[0], 0.0)

        e_R = 0.5 * vee(R_d.T @ R - R.T @ R_d)
        e_w = omega - R.T @ R_d @ self.w_d
        kR = g.att_wn**2
        kw = 2 * g.att_zeta * g.att_wn
        M_cmd = self.J * (-kR * e_R - kw * e_w) + np.cross(omega, self.J * omega) - M_a

        T_i = self.allocate(T_cmd, M_cmd, rotor_omega, v_air_b, omega, v_pack)
        thr, w_cmd = self.thrust_to_throttle(T_i, rotor_omega, v_air_b, omega, v_pack)
        diag = dict(
            F_des=F_des, a_cmd=a_cmd, R_d=R_d, alpha_d=alpha_d, T_cmd=T_cmd, M_cmd=M_cmd,
            T_i=T_i, w_cmd=w_cmd, e_R=e_R, alpha=aux["alpha"][0], V=aux["V"][0],
        )
        return thr, diag

    # ---------------------------------------------------------------- allocation
    def _hub_axial(self, v_air_b, omega):
        r = self.p.rotor_pos - self.cg
        return (v_air_b[None] + np.cross(omega[None], r)) @ self.p.rotor_axis

    def t_max(self, v_axial, v_pack):
        """Approximate steady-state max thrust per rotor at full throttle (no-load speed bound)."""
        w_max = 0.85 * v_pack * self.p.kv_rad  # loaded ~85 % of no-load
        T, _, _ = self.model.rotor_loads(np.full(4, w_max), v_axial)
        return T

    def allocate(self, T_cmd, M_cmd, rotor_omega, v_air_b, omega, v_pack):
        p = self.p
        r = p.rotor_pos - self.cg
        v_ax = self._hub_axial(v_air_b, omega)
        ct, cq, _ = self.model.rotor_coeffs(np.maximum(rotor_omega, 300.0), v_ax)
        kq = np.clip(cq * p.prop_diameter / np.maximum(ct, 1e-4), 0.005, 0.12)
        # rows: thrust, Mx, My, Mz ; with thrust along +x: M = r x (T,0,0) = (0, r_z T, -r_y T)
        B = np.stack([np.ones(4), -p.rotor_spin * kq, r[:, 2], -r[:, 1]], 0)
        Binv = np.linalg.inv(B)
        t_hi = self.t_max(v_ax, v_pack)
        W = np.array([T_cmd, *M_cmd])
        Ti = Binv @ W
        if np.all(Ti >= 0) and np.all(Ti <= t_hi):
            return Ti
        W[1] = np.clip(W[1], -0.3, 0.3)  # roll (reaction torque) is the weakest axis: shed first
        # attitude priority: keep the moment split, move the collective into the feasible band,
        # and only scale the moments down if no collective fits.
        per = Binv @ np.array([1.0, 0, 0, 0])  # per-rotor share of 1 N collective (all 0.25 here)
        dlt = Binv @ np.array([0, *W[1:]])
        c_lo = np.max(-dlt / per)
        c_hi = np.min((t_hi - dlt) / per)
        if c_lo <= c_hi:
            Ti = np.clip(T_cmd, c_lo, c_hi) * per + dlt
        else:
            # scale moments so that the band just closes: c_lo(s) = c_hi(s) is linear in s
            a0 = np.min(t_hi / per)
            a1 = np.max(-dlt / per) - np.min(-dlt / per)
            s_m = np.clip(a0 / max(a1, 1e-9), 0.0, 1.0)
            dlt = s_m * dlt
            c_lo, c_hi = np.max(-dlt / per), np.min((t_hi - dlt) / per)
            Ti = np.clip(T_cmd, c_lo, max(c_lo, c_hi)) * per + dlt
        return np.clip(Ti, 0.0, t_hi)

    def thrust_to_throttle(self, T_i, rotor_omega, v_air_b, omega, v_pack):
        p = self.p
        v_ax = np.clip(self._hub_axial(v_air_b, omega), 0, None)
        d = p.prop_diameter
        a = v_ax / (p.j0 * d)
        b = T_i / (p.rho * d**4 * p.ct0)
        n = 0.5 * (a + np.sqrt(a * a + 4 * b))
        w_cmd = 2 * np.pi * n
        # if T_i == 0 still keep the props turning a little (idle) so C_T(J) inversion stays sane
        w_cmd = np.maximum(w_cmd, 150.0)
        _, Q, _ = self.model.rotor_loads(w_cmd, v_ax)
        I = (Q + p.motor_viscous * w_cmd) / p.kt
        v_eff = I * p.r_phase + w_cmd / p.kv_rad + self.g.omega_p * (w_cmd - rotor_omega) / p.kv_rad
        thr = np.clip(v_eff / max(v_pack, 1.0), 0.0, 1.0)
        return thr, w_cmd
