"""Particle state representation and initialization."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np


class ParticleSystem:
    """Store particle state in an orthorhombic periodic simulation box."""

    def __init__(
        self,
        box: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray,
        charges: np.ndarray,
        sigmas: np.ndarray,
        epsilons: np.ndarray,
        names: Optional[Sequence[str]] = None,
    ):
        self.box = np.asarray(box, dtype=np.float64)
        self.positions = np.asarray(positions, dtype=np.float64)
        self.velocities = np.asarray(velocities, dtype=np.float64)
        self.masses = np.asarray(masses, dtype=np.float64)
        self.charges = np.asarray(charges, dtype=np.float64)
        self.sigmas = np.asarray(sigmas, dtype=np.float64)
        self.epsilons = np.asarray(epsilons, dtype=np.float64)

        self._validate_state()
        if names is None:
            self.names = [f"P_{i}" for i in range(self.n_particles)]
        else:
            self.names = [str(name) for name in names]
            if len(self.names) != self.n_particles:
                raise ValueError("names must contain one entry per particle.")

        self.forces = np.zeros_like(self.positions)
        self.apply_pbc()

    def _validate_state(self) -> None:
        if self.box.shape != (3,) or not np.all(np.isfinite(self.box)) or np.any(self.box <= 0):
            raise ValueError("box must contain three positive finite dimensions.")
        if self.positions.ndim != 2 or self.positions.shape[1:] != (3,):
            raise ValueError("positions must have shape (N, 3).")
        if self.positions.shape[0] == 0:
            raise ValueError("at least one particle is required.")
        if self.velocities.shape != self.positions.shape:
            raise ValueError("velocities must have the same shape as positions.")

        n = self.positions.shape[0]
        for name, values in (
            ("masses", self.masses),
            ("charges", self.charges),
            ("sigmas", self.sigmas),
            ("epsilons", self.epsilons),
        ):
            if values.shape != (n,):
                raise ValueError(f"{name} must have shape (N,).")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values.")
        if not np.all(np.isfinite(self.positions)) or not np.all(np.isfinite(self.velocities)):
            raise ValueError("positions and velocities must contain only finite values.")
        if np.any(self.masses <= 0):
            raise ValueError("masses must be positive.")
        if np.any(self.sigmas <= 0):
            raise ValueError("sigmas must be positive.")
        if np.any(self.epsilons < 0):
            raise ValueError("epsilons must be non-negative.")

    @property
    def n_particles(self) -> int:
        return int(self.positions.shape[0])

    def apply_pbc(self) -> None:
        """Wrap all particle coordinates into the primary cell ``[0, L)``."""
        self.positions = np.mod(self.positions, self.box)

    @property
    def kinetic_energy(self) -> float:
        """Return the total kinetic energy, ``1/2 sum(m v²)``."""
        return 0.5 * float(np.sum(self.masses[:, np.newaxis] * self.velocities**2))

    def temperature(self, kb: float = 1.0) -> float:
        """Return the instantaneous kinetic temperature after removing COM DOF."""
        kb = float(kb)
        if kb <= 0:
            raise ValueError("Boltzmann constant must be positive.")
        dof = 3.0 * (self.n_particles - 1)
        if dof <= 0:
            return 0.0
        return (2.0 * self.kinetic_energy) / (dof * kb)

    def remove_com_drift(self) -> None:
        """Remove the net center-of-mass linear momentum from all velocities."""
        total_mass = float(np.sum(self.masses))
        momentum = np.sum(self.masses[:, np.newaxis] * self.velocities, axis=0)
        self.velocities -= momentum / total_mass

    def initialize_velocities(self, target_temp: float, kb: float = 1.0) -> None:
        """Assign Maxwell-Boltzmann velocities rescaled to ``target_temp``."""
        target_temp = float(target_temp)
        kb = float(kb)
        if target_temp < 0 or kb <= 0:
            raise ValueError("target_temp must be non-negative and kb must be positive.")
        if target_temp == 0:
            self.velocities.fill(0.0)
            return

        std = np.sqrt(kb * target_temp / self.masses)[:, np.newaxis]
        self.velocities = np.random.normal(0.0, 1.0, size=self.positions.shape) * std
        self.remove_com_drift()

        # Rescale exactly to the requested temperature (up to floating-point precision).
        current_temp = self.temperature(kb)
        if current_temp > 1e-14:
            self.velocities *= np.sqrt(target_temp / current_temp)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "ParticleSystem":
        """Construct and initialize a system from a validated configuration."""
        system_cfg = cfg["system"]
        box = np.array(system_cfg["box_size"], dtype=np.float64)
        species_list = system_cfg.get("species", [])
        if not species_list:
            raise ValueError("No species defined in system configuration.")

        masses = []
        charges = []
        sigmas = []
        epsilons = []
        names = []
        for species in species_list:
            count = int(species.get("count", 1))
            for _ in range(count):
                masses.append(float(species.get("mass", 1.0)))
                charges.append(float(species.get("charge", 0.0)))
                sigmas.append(float(species.get("sigma", 1.0)))
                epsilons.append(float(species.get("epsilon", 1.0)))
                names.append(str(species.get("name", "Atom")))

        n_particles = len(masses)
        positions = cls._place_particles(
            n_particles,
            box,
            mode=str(system_cfg.get("lattice", "grid")).lower(),
        )
        system = cls(
            box=box,
            positions=positions,
            velocities=np.zeros_like(positions),
            masses=np.array(masses, dtype=np.float64),
            charges=np.array(charges, dtype=np.float64),
            sigmas=np.array(sigmas, dtype=np.float64),
            epsilons=np.array(epsilons, dtype=np.float64),
            names=names,
        )
        system.initialize_velocities(
            float(system_cfg.get("initial_temperature", 1.0)),
            kb=float(system_cfg.get("boltzmann_constant", 1.0)),
        )
        return system

    @staticmethod
    def _place_particles(n: int, box: np.ndarray, mode: str = "grid") -> np.ndarray:
        """Place particles on a grid or randomly with a minimum periodic separation."""
        if n < 1:
            raise ValueError("at least one particle is required.")
        box = np.asarray(box, dtype=np.float64)

        if mode == "grid":
            n_per_axis = int(np.ceil(n ** (1.0 / 3.0)))
            spacings = box / n_per_axis
            axes = [
                np.linspace(spacing * 0.5, length - spacing * 0.5, n_per_axis)
                for length, spacing in zip(box, spacings)
            ]
            coords = []
            for x in axes[0]:
                for y in axes[1]:
                    for z in axes[2]:
                        if len(coords) == n:
                            break
                        coords.append([x, y, z])
                    if len(coords) == n:
                        break
                if len(coords) == n:
                    break
            return np.asarray(coords, dtype=np.float64)

        if mode == "random":
            points = []
            # Keep the requested default separation, but do not make a box smaller
            # than that separation impossible by construction.
            min_dist = min(0.7, float(np.min(box)) * 0.5)
            for _ in range(10000):
                candidate = np.random.uniform(0.0, box)
                if not points:
                    points.append(candidate)
                else:
                    displacement = np.abs(candidate - np.asarray(points))
                    displacement = np.minimum(displacement, box - displacement)
                    distances = np.sqrt(np.sum(displacement**2, axis=1))
                    if np.all(distances >= min_dist):
                        points.append(candidate)
                if len(points) == n:
                    return np.asarray(points, dtype=np.float64)
            raise RuntimeError("Could not place particles without overlap. Try 'grid' mode.")

        raise ValueError(f"Unknown placement mode: {mode}")
