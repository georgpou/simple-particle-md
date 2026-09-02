"""Velocity-Verlet integration and simple NVT thermostats."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class Integrator(ABC):
    """Abstract interface for time integration schemes."""

    @abstractmethod
    def step(self, system, potential) -> Tuple[float, float, float]:
        """Advance a system by one step and return potential energy components."""


class VelocityVerlet(Integrator):
    """Velocity-Verlet integrator supporting NVE and NVT ensembles."""

    def __init__(
        self,
        dt: float = 0.002,
        ensemble: str = "NVT",
        thermostat: str = "berendsen",
        target_temperature: float = 1.0,
        tau: float = 0.05,
        collision_frequency: float = 1.0,
        kb: float = 1.0,
    ):
        self.dt = float(dt)
        self.ensemble = str(ensemble).upper()
        self.thermostat = str(thermostat).lower() if thermostat else "berendsen"
        self.target_temp = float(target_temperature)
        self.tau = float(tau)
        self.collision_freq = float(collision_frequency)
        self.kb = float(kb)

        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be positive and finite.")
        if self.ensemble not in {"NVE", "NVT"}:
            raise ValueError("ensemble must be 'NVE' or 'NVT'.")
        if self.thermostat not in {"berendsen", "andersen"}:
            raise ValueError("thermostat must be 'berendsen' or 'andersen'.")
        if not np.isfinite(self.target_temp) or self.target_temp < 0:
            raise ValueError("target_temperature must be non-negative and finite.")
        if not np.isfinite(self.tau) or self.tau <= 0:
            raise ValueError("tau must be positive and finite.")
        if not np.isfinite(self.collision_freq) or self.collision_freq < 0:
            raise ValueError("collision_frequency must be non-negative and finite.")
        if not np.isfinite(self.kb) or self.kb <= 0:
            raise ValueError("kb must be positive and finite.")

    def step(self, system, potential) -> Tuple[float, float, float]:
        """Advance ``system`` by one Velocity-Verlet step."""
        inv_masses = 1.0 / system.masses[:, np.newaxis]

        system.velocities += 0.5 * system.forces * inv_masses * self.dt
        system.positions += system.velocities * self.dt
        system.apply_pbc()

        forces, epot, elj, ecoul = potential.compute(system)
        system.forces = forces
        system.velocities += 0.5 * system.forces * inv_masses * self.dt

        if self.ensemble == "NVT":
            self._apply_thermostat(system)
        return epot, elj, ecoul

    def _apply_thermostat(self, system) -> None:
        """Regulate kinetic temperature using the configured NVT thermostat."""
        current_temp = system.temperature(self.kb)

        if self.thermostat == "berendsen":
            if current_temp > 1e-14:
                scaling_sq = 1.0 + (self.dt / self.tau) * (
                    self.target_temp / current_temp - 1.0
                )
                if scaling_sq > 0:
                    scale = np.sqrt(scaling_sq)
                    # Avoid a single unstable step causing a numerical shock.
                    system.velocities *= np.clip(scale, 0.7, 1.4)
            elif self.target_temp > 0:
                # A cold system can occur when starting from zero velocities.
                system.initialize_velocities(self.target_temp, kb=self.kb)

        elif self.thermostat == "andersen":
            probability = min(1.0, self.collision_freq * self.dt)
            collision_mask = np.random.random(system.n_particles) < probability
            if np.any(collision_mask):
                std = np.sqrt(self.kb * self.target_temp / system.masses[collision_mask])
                system.velocities[collision_mask] = (
                    np.random.normal(0.0, 1.0, size=(int(np.sum(collision_mask)), 3))
                    * std[:, np.newaxis]
                )
