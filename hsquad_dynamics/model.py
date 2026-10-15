"""Force model for the Bullet airframe (guide §5).

Single entry point:

    F_body, M_body, out = model.step(state, motor_cmd, wind, dt)

Everything is batched over a leading dimension N (N = 1 for a single vehicle, N = num_envs for
Isaac Lab). Written against NumPy; the Isaac Sim 6.1 Python ships without PyTorch, and every
operation used here has a 1:1 torch equivalent.

Frames: body = FLU, world = Isaac Z-up. F_body and M_body are in body axes, M_body about
``ref_point`` (default: base_link CG, which is where the sim applies the wrench).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import BulletParams


@dataclass
class State:
    R_wb: np.ndarray  # (N,3,3) body -> world rotation
    v_world: np.ndarray  # (N,3)
    omega_body: np.ndarray  # (N,3) body rates, FLU
    rotor_omega: np.ndarray  # (N,4) rad/s, >= 0 (sense given by params.rotor_spin)
    soc: np.ndarray  # (N,)
    v_rc: np.ndarray  # (N,) battery RC-branch voltage

    @staticmethod
    def initial(n: int = 1, soc: float = 1.0) -> "State":
        return State(
            R_wb=np.tile(np.eye(3), (n, 1, 1)),
            v_world=np.zeros((n, 3)),
            omega_body=np.zeros((n, 3)),
            rotor_omega=np.zeros((n, 4)),
            soc=np.full((n,), soc),
            v_rc=np.zeros((n,)),
        )


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """np.cross for (..., 3) arrays without its per-call axis bookkeeping (it dominates at N = 1)."""
    a, b = np.broadcast_arrays(a, b)
    out = np.empty(a.shape)
    out[..., 0] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    out[..., 1] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    out[..., 2] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return out


def _stall_blend(alpha: np.ndarray, a0: float, m: float) -> np.ndarray:
    """Beard & McLain sigmoid: 0 in attached flow, 1 past +-a0."""
    ea = np.exp(np.clip(-m * (alpha - a0), -50, 50))
    eb = np.exp(np.clip(m * (alpha + a0), -50, 50))
    return (1.0 + ea + eb) / ((1.0 + ea) * (1.0 + eb))


class BulletDynamics:
    def __init__(self, params: BulletParams | None = None, ref_point: np.ndarray | None = None):
        self.p = params or BulletParams()
        self.ref = self.p.cg_body.copy() if ref_point is None else np.asarray(ref_point, dtype=float)

    # ------------------------------------------------------------------ rotor
    def rotor_coeffs(self, omega: np.ndarray, v_axial: np.ndarray):
        """C_T, C_Q and advance ratio J for rotor speeds omega (rad/s) and axial inflow (m/s)."""
        p = self.p
        n = np.maximum(omega / (2 * np.pi), 1.0)  # rev/s, floored to keep J finite at spin-up
        J = np.maximum(v_axial, 0.0) / (n * p.prop_diameter)
        ct = np.maximum(p.ct0 * (1.0 - J / p.j0), 0.0)
        cq = np.maximum(p.cq0 * (1.0 - J / p.jq0), 0.0)
        return ct, cq, J

    def rotor_loads(self, omega: np.ndarray, v_axial: np.ndarray):
        p = self.p
        n = omega / (2 * np.pi)
        ct, cq, J = self.rotor_coeffs(omega, v_axial)
        T = ct * p.rho * n**2 * p.prop_diameter**4
        Q = cq * p.rho * n**2 * p.prop_diameter**5
        return T, Q, J

    # ------------------------------------------------------------------ battery
    def ocv(self, soc: np.ndarray) -> np.ndarray:
        return self.p.cells * np.interp(np.clip(soc, 0, 1), self.p.ocv_soc, self.p.ocv_cell)

    # ------------------------------------------------------------------ aero
    def aero(self, v_air_body: np.ndarray, omega_body: np.ndarray):
        """Wing + fuselage aerodynamic force/moment, body axes, moment about self.ref."""
        p = self.p
        N = v_air_body.shape[0]
        vx, vy, vz = v_air_body[:, 0], v_air_body[:, 1], v_air_body[:, 2]
        V = np.linalg.norm(v_air_body, axis=1)
        Vs = np.maximum(V, 1e-3)
        alpha = np.arctan2(-vz, vx)  # + nose up (guide §5.1)
        beta = np.arcsin(np.clip(vy / Vs, -1, 1))
        q = 0.5 * p.rho * V**2
        vhat = v_air_body / Vs[:, None]
        e_lift = np.stack([np.sin(alpha), np.zeros(N), np.cos(alpha)], 1)  # perp. to flow in x-z plane

        # wing (guide §5.3), blended to flat-plate past stall so the full transition envelope is covered
        sig = _stall_blend(alpha, p.alpha_stall, p.stall_sharpness)
        sa, ca = np.sin(alpha), np.cos(alpha)
        cl_att = p.cl_alpha * alpha
        cl = (1 - sig) * cl_att + sig * 2.0 * np.sign(alpha) * sa**2 * ca
        cd_att = p.cd0_wing + cl_att**2 / (np.pi * p.oswald * p.aspect_ratio)
        cd = (1 - sig) * cd_att + sig * (p.cd0_wing + 2.0 * sa**2)
        cm = (1 - sig) * (p.cm0 + p.cm_alpha * alpha) + sig * (-0.5 * sa)  # flat plate: CP at mid-chord
        # wing area projected on the flow in the x-z plane only (no span-wise flow contribution)
        q_xz = 0.5 * p.rho * (vx**2 + vz**2)
        F_w = (q_xz * p.wing_area)[:, None] * (cl[:, None] * e_lift) - (q * p.wing_area * cd)[:, None] * vhat
        span = np.sqrt(p.aspect_ratio * p.wing_area)
        pr, qr, rr = omega_body[:, 0], omega_body[:, 1], omega_body[:, 2]
        two_v = 2 * Vs
        M_w = np.stack(
            [
                q * p.wing_area * span * p.cl_p * pr * span / two_v,
                # Cm + nose-up == rotation about -y in FLU
                -q * p.wing_area * p.mac * cm + q * p.wing_area * p.mac * p.cm_q * qr * p.mac / two_v,
                q * p.wing_area * span * p.cn_r * rr * span / two_v,
            ],
            1,
        )
        M_w = M_w + _cross(p.wing_c4 - self.ref, F_w)

        # fuselage (guide §5.4)
        cn_b = p.cn_alpha_body * 0.5 * np.sin(2 * alpha)  # = CNa*alpha for small alpha, symmetric at 90 deg
        cy_b = -p.cn_alpha_body * 0.5 * np.sin(2 * beta)
        F_b = (q * p.body_area)[:, None] * (
            np.stack([np.zeros(N), cy_b, cn_b], 1) - p.cd0_body * vhat
        )
        xcp = np.interp(np.degrees(np.abs(alpha)), p.xcp_alpha_deg, p.xcp_body)
        r_cp = np.stack([xcp, np.zeros(N), np.zeros(N)], 1) - self.ref
        M_b = _cross(r_cp, F_b)

        aux = dict(V=V, alpha=alpha, beta=beta, q=q, F_wing=F_w, F_body=F_b)
        return F_w + F_b, M_w + M_b, aux

    # ------------------------------------------------------------------ electrical only
    def motor_electrical(self, s: State, motor_cmd: np.ndarray, omega: np.ndarray, dt: float):
        """ESC + motor + battery for one step at externally integrated rotor speeds.

        For simulators that integrate the rotor speed themselves (PhysX joints): takes the rotor
        speeds omega (N,4), returns the electromagnetic motor torque minus bearing loss (N,4) and
        the electrical diagnostics. Prop torque is NOT included. Updates s.soc and s.v_rc.
        Same equations as the sub-loop in step(), except that motor current is clamped at >= 0
        (no active braking), so spin-down comes from prop drag alone.
        """
        p = self.p
        thr = np.clip(motor_cmd, 0.0, 1.0)
        omega = np.maximum(omega, 0.0)
        v_oc = self.ocv(s.soc) - s.v_rc
        emf = omega / p.kv_rad
        v_pack = (v_oc + p.r0 * np.sum(thr * emf, 1) / p.r_phase) / (1.0 + p.r0 * np.sum(thr * thr, 1) / p.r_phase)
        I_m = (thr * v_pack[:, None] - emf) / p.r_phase
        if np.any(np.abs(I_m) > p.esc_current_limit):
            lim = np.abs(I_m) > p.esc_current_limit
            I_fix = np.where(lim, np.sign(I_m) * p.esc_current_limit, 0.0)
            tf = np.where(lim, 0.0, thr)
            v_pack = (v_oc - p.r0 * np.sum(thr * I_fix, 1) + p.r0 * np.sum(tf * emf, 1) / p.r_phase) / (
                1.0 + p.r0 * np.sum(tf * tf, 1) / p.r_phase
            )
            I_m = np.where(lim, I_fix, (thr * v_pack[:, None] - emf) / p.r_phase)
        # a PWM ESC without active braking cannot sink current from a spinning motor
        I_m = np.maximum(I_m, 0.0)
        v_pack = np.maximum(v_pack, 0.0)
        I_bat = np.sum((I_m * p.r_phase + emf) * I_m, 1) / np.maximum(v_pack, 1.0)
        s.v_rc = s.v_rc + dt * (I_bat / p.c1 - s.v_rc / (p.r1 * p.c1))
        s.soc = np.minimum(np.maximum(s.soc - dt * I_bat / (p.capacity_ah * 3600.0), 0.0), 1.0)
        tau = p.kt * I_m - p.motor_viscous * omega
        return tau, dict(current=I_m, i_bat=I_bat, v_pack=v_pack)

    # ------------------------------------------------------------------ full step
    def step(self, s: State, motor_cmd: np.ndarray, wind: np.ndarray | None, dt: float):
        """Advance motors/battery by dt and return the wrench to apply over that step.

        motor_cmd: (N,4) throttle in [0, 1].
        Returns F_body (N,3), M_body (N,3) about self.ref, and a dict with the new rotor speeds,
        battery state and diagnostics. ``s`` is updated in place (rotor_omega, soc, v_rc).
        """
        p = self.p
        N = s.v_world.shape[0]
        wind = np.zeros((N, 3)) if wind is None else np.broadcast_to(wind, (N, 3))
        v_air_body = np.einsum("nji,nj->ni", s.R_wb, s.v_world - wind)  # R^T (v - w)
        # rotor inflow includes the rotation-induced velocity at each hub
        r_hub = p.rotor_pos[None] - self.ref  # (1,4,3)
        v_hub = v_air_body[:, None, :] + _cross(s.omega_body[:, None, :], r_hub)
        v_axial = v_hub @ p.rotor_axis

        thr = np.clip(motor_cmd, 0.0, 1.0)
        h = dt / p.motor_substeps
        omega = s.rotor_omega.copy()
        T_acc = np.zeros((N, 4))
        Q_acc = np.zeros((N, 4))
        I_acc = np.zeros((N, 4))
        Ib_acc = np.zeros(N)
        Vp_acc = np.zeros(N)
        for _ in range(p.motor_substeps):
            # ideal PWM ESC with a per-motor current limit; battery current from power balance
            # (V_pack I_bat = sum V_eff I_m). The pack sag is solved implicitly: an explicit
            # fixed point has loop gain 4 R0 / R_phase > 1 and oscillates.
            v_oc = self.ocv(s.soc) - s.v_rc
            emf = omega / p.kv_rad
            v_pack = (v_oc + p.r0 * np.sum(thr * emf, 1) / p.r_phase) / (1.0 + p.r0 * np.sum(thr * thr, 1) / p.r_phase)
            I_m = (thr * v_pack[:, None] - emf) / p.r_phase
            if np.any(np.abs(I_m) > p.esc_current_limit):
                # limited motors draw a fixed current: re-solve with only the unlimited ones in the sag term
                lim = np.abs(I_m) > p.esc_current_limit
                I_fix = np.where(lim, np.sign(I_m) * p.esc_current_limit, 0.0)
                tf = np.where(lim, 0.0, thr)
                v_pack = (v_oc - p.r0 * np.sum(thr * I_fix, 1) + p.r0 * np.sum(tf * emf, 1) / p.r_phase) / (
                    1.0 + p.r0 * np.sum(tf * tf, 1) / p.r_phase
                )
                I_m = np.where(lim, I_fix, (thr * v_pack[:, None] - emf) / p.r_phase)
            v_pack = np.maximum(v_pack, 0.0)
            I_bat = np.sum((I_m * p.r_phase + emf) * I_m, 1) / np.maximum(v_pack, 1.0)
            T, Q, _ = self.rotor_loads(omega, v_axial)
            tau = p.kt * I_m - Q - p.motor_viscous * omega
            omega = np.maximum(omega + h * tau / p.rotor_inertia, 0.0)
            s.v_rc = s.v_rc + h * (I_bat / p.c1 - s.v_rc / (p.r1 * p.c1))
            s.soc = np.minimum(np.maximum(s.soc - h * I_bat / (p.capacity_ah * 3600.0), 0.0), 1.0)
            T_acc += T
            Q_acc += Q
            I_acc += I_m
            Ib_acc += I_bat
            Vp_acc += v_pack
        s.rotor_omega = omega
        k = 1.0 / p.motor_substeps
        T, Q = T_acc * k, Q_acc * k

        # propulsion wrench (guide §5.2)
        a = p.rotor_axis
        F_rot = T[:, :, None] * a  # (N,4,3)
        M_rot = _cross(r_hub, F_rot) - (p.rotor_spin * Q)[:, :, None] * a
        F_prop, M_prop = F_rot.sum(1), M_rot.sum(1)

        F_aero, M_aero, aux = self.aero(v_air_body, s.omega_body)
        out = dict(
            thrust=T,
            torque=Q,
            current=I_acc * k,
            i_bat=Ib_acc * k,
            v_pack=Vp_acc * k,
            rotor_omega=omega,
            F_prop=F_prop,
            M_prop=M_prop,
            F_aero=F_aero,
            M_aero=M_aero,
            v_air_body=v_air_body,
            **aux,
        )
        return F_prop + F_aero, M_prop + M_aero, out
