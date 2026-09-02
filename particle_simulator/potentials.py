"""Vectorized Lennard-Jones and Coulomb interactions with periodic boundaries."""

from __future__ import annotations

from typing import Tuple

import numpy as np


class LennardJonesCoulombPotential:
    """Compute pairwise VDW and electrostatic energies and forces.

    Lennard-Jones parameters use Lorentz-Berthelot combining rules. Coulomb and
    Lennard-Jones energies are truncated at ``cutoff_radius`` and optionally
    shifted so their values are zero at the cutoff.
    """

    def __init__(
        self,
        cutoff_radius: float = 4.0,
        coulomb_constant: float = 1.0,
        shift: bool = True,
    ):
        self.r_cut = float(cutoff_radius)
        self.ke = float(coulomb_constant)
        self.shift = bool(shift)
        if not np.isfinite(self.r_cut) or self.r_cut <= 0:
            raise ValueError("cutoff_radius must be positive and finite.")
        if not np.isfinite(self.ke) or self.ke < 0:
            raise ValueError("coulomb_constant must be non-negative and finite.")

    def compute(self, system) -> Tuple[np.ndarray, float, float, float]:
        """Return forces, total potential energy, LJ energy, and Coulomb energy."""
        pos = np.asarray(system.positions, dtype=np.float64)
        box = np.asarray(system.box, dtype=np.float64)
        sigma = np.asarray(system.sigmas, dtype=np.float64)
        epsilon = np.asarray(system.epsilons, dtype=np.float64)
        charges = np.asarray(system.charges, dtype=np.float64)
        n_particles = pos.shape[0]

        if pos.shape != (n_particles, 3) or box.shape != (3,):
            raise ValueError("system positions and box have invalid shapes.")
        if self.r_cut > float(np.min(box)) / 2.0:
            raise ValueError("cutoff_radius must be <= half the smallest box dimension.")

        # rij[i, j] points from particle i to particle j.
        rij = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
        rij -= box * np.round(rij / box)
        r2 = np.sum(rij**2, axis=-1)
        np.fill_diagonal(r2, np.inf)

        if np.any(np.isfinite(r2) & (r2 <= np.finfo(np.float64).eps)):
            raise ValueError("Particle overlap detected; pair potential is singular at r=0.")

        mask = r2 < self.r_cut**2
        r = np.sqrt(np.where(mask, r2, 1.0))
        rinv = np.where(mask, 1.0 / r, 0.0)
        rinv2 = rinv**2

        sig_ij = 0.5 * (sigma[:, np.newaxis] + sigma[np.newaxis, :])
        eps_ij = np.sqrt(epsilon[:, np.newaxis] * epsilon[np.newaxis, :])
        q_ij = charges[:, np.newaxis] * charges[np.newaxis, :]

        sr6 = (sig_ij * rinv) ** 6
        sr12 = sr6**2

        # The force on i is (dV/dr)/r * (r_j - r_i).
        dv_dr_r_lj = np.where(
            mask,
            -24.0 * eps_ij * rinv2 * (2.0 * sr12 - sr6),
            0.0,
        )
        dv_dr_r_coul = np.where(
            mask,
            -self.ke * q_ij * (rinv2 * rinv),
            0.0,
        )
        forces = np.sum(
            (dv_dr_r_lj + dv_dr_r_coul)[..., np.newaxis] * rij,
            axis=1,
        )

        v_lj = 4.0 * eps_ij * (sr12 - sr6)
        v_coul = self.ke * q_ij * rinv
        if self.shift:
            rc = self.r_cut
            sr6_cut = (sig_ij / rc) ** 6
            sr12_cut = sr6_cut**2
            v_lj -= 4.0 * eps_ij * (sr12_cut - sr6_cut)
            v_coul -= self.ke * q_ij / rc

        v_lj = np.where(mask, v_lj, 0.0)
        v_coul = np.where(mask, v_coul, 0.0)
        e_lj = 0.5 * float(np.sum(v_lj))
        e_coul = 0.5 * float(np.sum(v_coul))
        return forces, e_lj + e_coul, e_lj, e_coul
