"""Simulation orchestration and trajectory persistence."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


class Simulation:
    """Execute a molecular-dynamics trajectory and record thermodynamics."""

    def __init__(self, system, potential, integrator, config: Dict[str, Any]):
        self.system = system
        self.potential = potential
        self.integrator = integrator
        self.config = config

        simulation_cfg = config["simulation"]
        self.total_steps = int(simulation_cfg.get("total_steps", 1000))
        self.log_interval = int(simulation_cfg.get("log_interval", 20))
        self.traj_interval = int(simulation_cfg.get("trajectory_interval", 10))
        if self.total_steps < 0:
            raise ValueError("total_steps must be non-negative.")
        if self.log_interval <= 0 or self.traj_interval <= 0:
            raise ValueError("log_interval and trajectory_interval must be positive.")

        self.output_cfg = config.get("output", {})
        self.out_dir = Path(self.output_cfg.get("output_dir", "./sim_results"))
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.trajectory: List[np.ndarray] = []
        self.history: List[Dict[str, float]] = []

    def run(self) -> None:
        """Execute the configured simulation and save numerical results."""
        # Re-running an instance starts a fresh trajectory.
        self.trajectory.clear()
        self.history.clear()

        forces, epot, elj, ecoul = self.potential.compute(self.system)
        self.system.forces = forces
        kb = float(self.config["system"].get("boltzmann_constant", 1.0))

        print("=== Starting MD Simulation ===")
        print(f"Particles: {self.system.n_particles} | Box: {self.system.box}")
        print(f"Ensemble: {self.integrator.ensemble} | Timestep dt: {self.integrator.dt}")
        print(f"Steps: {self.total_steps} | Output dir: {self.out_dir}\n")

        header = (
            f"{'Step':>7} | {'Time':>8} | {'E_pot':>10} | {'E_kin':>10} | "
            f"{'E_tot':>10} | {'Temp':>8}"
        )
        print(header)
        print("-" * len(header))

        for step in range(self.total_steps + 1):
            if step > 0:
                epot, elj, ecoul = self.integrator.step(self.system, self.potential)

            if step % self.traj_interval == 0:
                self.trajectory.append(self.system.positions.copy())

            if step % self.log_interval == 0:
                kinetic = self.system.kinetic_energy
                temperature = self.system.temperature(kb)
                total_energy = kinetic + epot
                time_value = step * self.integrator.dt
                entry: Dict[str, float] = {
                    "step": step,
                    "time": time_value,
                    "e_pot": epot,
                    "e_lj": elj,
                    "e_coul": ecoul,
                    "e_kin": kinetic,
                    "e_tot": total_energy,
                    "temperature": temperature,
                }
                self.history.append(entry)

                if step % (self.log_interval * 5) == 0 or step == self.total_steps:
                    print(
                        f"{step:7d} | {time_value:8.3f} | {epot:10.3f} | "
                        f"{kinetic:10.3f} | {total_energy:10.3f} | {temperature:8.3f}"
                    )

        print("-" * len(header))
        print("Simulation complete! Saving results...\n")
        self.save_data()

    def save_data(self) -> None:
        """Save thermodynamic CSV data and a compressed trajectory archive."""
        log_path = self.out_dir / self.output_cfg.get("log_file", "thermodynamics.csv")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.history:
            keys = list(self.history[0].keys())
            with log_path.open("w", newline="", encoding="utf-8") as log_file:
                writer = csv.DictWriter(log_file, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.history)
            print(f"[+] Thermodynamic logs saved: {log_path}")

        trajectory_path = self.out_dir / self.output_cfg.get("trajectory_file", "trajectory.npz")
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        if self.trajectory:
            trajectory = np.asarray(self.trajectory, dtype=np.float64)
        else:
            trajectory = np.empty((0, self.system.n_particles, 3), dtype=np.float64)
        np.savez_compressed(
            trajectory_path,
            trajectory=trajectory,
            box=self.system.box,
            charges=self.system.charges,
            sigmas=self.system.sigmas,
            epsilons=self.system.epsilons,
            masses=self.system.masses,
            names=np.asarray(self.system.names),
        )
        print(f"[+] Trajectory saved: {trajectory_path}")
