"""Vehicle parameters for the Bullet airframe (3.3626 kg configuration), FLU body frame.

FLU = x forward, y port, z up. Positions are measured from the nose apex (the base_link origin of
assets/bullet_flu.urdf), so the CG is at x = -0.2871 (spec sheet -0.3071, moved 20 mm forward in URDF rev e).

Anything tagged PLACEHOLDER is not on the CFD spec sheet (guide §9) and must be replaced when the
engineer supplies it. Every placeholder lives here, so domain randomisation and later data drops
only touch this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BulletParams:
    # ---------------------------------------------------------------- environment
    rho: float = 1.225  # kg/m^3, ISA sea level
    g: float = 9.81

    # ---------------------------------------------------------------- mass properties (spec sheet)
    mass_body: float = 3.3626  # base_link, assumption A3: excludes props
    mass_rotor: float = 0.021
    # URDF rev 2026-10-14e: 20 mm forward of the spec-sheet -0.3071 for static pitch stability
    cg_body: np.ndarray = field(default_factory=lambda: np.array([-0.2871, 0.0, 0.0]))
    inertia_body: np.ndarray = field(default_factory=lambda: np.array([0.0496, 0.1037, 0.1031]))  # URDF rev d

    # ---------------------------------------------------------------- rotors (URDF joints, FLU)
    rotor_pos: np.ndarray = field(
        default_factory=lambda: np.array(
            [  # FLU, from assets/bullet_flu.urdf (URDF rev 2026-10-14d: airframe rolled so the wing spans y)
                [-0.548222132, 0.137732767, 0.137732823],  # rotor_0, handedness A
                [-0.548222132, 0.137732767, -0.137732823],  # rotor_1, handedness B
                [-0.548222132, -0.137732767, 0.137732823],  # rotor_2, handedness B
                [-0.548222132, -0.137732767, -0.137732823],  # rotor_3, handedness A
            ]
        )
    )
    rotor_axis: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))  # thrust along +x (forward)
    # Spin sense about +x (thrust direction): from the CAD blade pitch in meshes/visual/propeller_*.stl
    # (tools/check_urdf_vs_cad.py: 23 deg section pitch at r = 75 mm, all sections agree): rotor_0/3 = -1,
    # rotor_1/2 = +1. The earlier placeholder had all four reversed; with the wing-horizontal URDF (rev
    # 2026-10-14d) that reversed set made PX4's fixed-wing roll/yaw loop oscillate (bank +/-30 deg), the CAD
    # set holds +/-1 deg. PX4 CA_ROTORn_KM signs in px4/airframes/22001_isaac_bullet must match.
    rotor_spin: np.ndarray = field(default_factory=lambda: np.array([-1.0, 1.0, 1.0, -1.0]))
    prop_diameter: float = 0.2275  # m, from the URDF rotor collision disc (r = 0.11375)
    rotor_inertia: float = 4.746789149e-05  # kg m^2 about the spin axis (URDF ixx)

    # Linear advance-ratio fallback C_T = C_T0 (1 - J/J0), guide §5.2. PLACEHOLDER until BEMT tables exist.
    ct0: float = 0.12
    j0: float = 1.4
    cq0: float = 0.0085  # PLACEHOLDER, C_P0 = 2 pi C_Q0 ~ 0.053
    jq0: float = 1.6  # PLACEHOLDER, torque goes to zero a little after thrust (windmill onset)

    # ---------------------------------------------------------------- motor / ESC (guide §5.6)
    kv_rpm: float = 700.0
    r_phase: float = 0.0406  # ohm, Y-equivalent
    motor_viscous: float = 2.0e-6  # N m s/rad, PLACEHOLDER bearing/eddy loss
    esc_current_limit: float = 65.0  # A per motor, PLACEHOLDER (ESC rating)
    motor_substeps: int = 5  # 5 kHz electrical/mechanical sub-loop at 1 kHz physics

    # ---------------------------------------------------------------- battery, Tattu 5100 6S, Thevenin
    cells: int = 6
    capacity_ah: float = 5.1
    r0: float = 0.012  # ohm, PLACEHOLDER pack DC resistance
    r1: float = 0.006  # ohm, PLACEHOLDER RC branch
    c1: float = 2000.0  # F, PLACEHOLDER RC branch
    ocv_soc: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]))
    ocv_cell: np.ndarray = field(default_factory=lambda: np.array([3.30, 3.55, 3.65, 3.72, 3.80, 3.88, 4.00, 4.08, 4.20]))

    # ---------------------------------------------------------------- wing (spec sheet + placeholders)
    wing_area: float = 0.1094009
    wing_c4: np.ndarray = field(default_factory=lambda: np.array([-0.2776, 0.0, 0.0]))
    cl_alpha: float = 3.621  # /rad
    cd0_wing: float = 0.052792
    cm_alpha: float = -0.6  # /rad, about c/4, on wing area
    cm0: float = 0.0  # PLACEHOLDER (engineer)
    mac: float = 0.15  # m, PLACEHOLDER mean aerodynamic chord (engineer)
    aspect_ratio: float = 4.86  # PLACEHOLDER (span 0.73 m at the placeholder chord)
    oswald: float = 0.85  # PLACEHOLDER
    alpha_stall: float = 0.26  # rad, 15 deg PLACEHOLDER
    stall_sharpness: float = 50.0  # blending-sigmoid rate
    # Rate damping on wing area / chord. PLACEHOLDER (no dynamic derivatives on the spec sheet).
    cl_p: float = -0.40
    cm_q: float = -6.0
    cn_r: float = -0.15

    # ---------------------------------------------------------------- fuselage (spec sheet + placeholders)
    body_area: float = 0.0145734  # frontal
    cd0_body: float = 0.0070325
    cn_alpha_body: float = 2.0  # /rad slender body, PLACEHOLDER
    # CP vs |alpha| (deg), FLU x from nose. Only 0 and 2 deg are on the sheet; held constant beyond 2 deg.
    xcp_alpha_deg: np.ndarray = field(default_factory=lambda: np.array([0.0, 2.0]))
    xcp_body: np.ndarray = field(default_factory=lambda: np.array([-0.392, -0.329]))

    # ---------------------------------------------------------------- derived
    @property
    def mass_total(self) -> float:
        return self.mass_body + 4 * self.mass_rotor

    @property
    def cg_total(self) -> np.ndarray:
        return (self.mass_body * self.cg_body + self.mass_rotor * self.rotor_pos.sum(0)) / self.mass_total

    @property
    def inertia_total(self) -> np.ndarray:
        """Diagonal inertia of the full vehicle about cg_total (parallel axis, rotors as point masses + own)."""
        d = self.cg_body - self.cg_total
        i = self.inertia_body + self.mass_body * np.array([d[1] ** 2 + d[2] ** 2, d[0] ** 2 + d[2] ** 2, d[0] ** 2 + d[1] ** 2])
        for p in self.rotor_pos:
            r = p - self.cg_total
            i = i + self.mass_rotor * np.array([r[1] ** 2 + r[2] ** 2, r[0] ** 2 + r[2] ** 2, r[0] ** 2 + r[1] ** 2])
            i = i + np.array([self.rotor_inertia, 4.7686e-05, 4.7686e-05])
        return i

    @property
    def kv_rad(self) -> float:
        """Back-EMF constant in rad/s per volt."""
        return self.kv_rpm * 2.0 * np.pi / 60.0

    @property
    def kt(self) -> float:
        return 1.0 / self.kv_rad  # N m / A (SI: Kt = Ke)
