"""Command-line interface for running a configured simulation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .config import load_config
from .integrators import VelocityVerlet
from .potentials import LennardJonesCoulombPotential
from .simulation import Simulation
from .system import ParticleSystem


def run_cli(argv: Optional[Sequence[str]] = None) -> int:
    """Parse command-line arguments, run the simulation, and create outputs."""
    parser = argparse.ArgumentParser(
        description="Vectorized periodic molecular dynamics with VDW and electrostatics"
    )
    parser.add_argument("config", type=str, help="Path to the YAML simulation configuration file")
    parser.add_argument("--no-animation", action="store_true", help="Skip 3D animation generation")
    parser.add_argument("--no-plots", action="store_true", help="Skip thermodynamic and RDF plots")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    system = ParticleSystem.from_config(cfg)

    potential_cfg = cfg["potential"]
    potential = LennardJonesCoulombPotential(
        cutoff_radius=potential_cfg["cutoff_radius"],
        coulomb_constant=potential_cfg["coulomb_constant"],
        shift=potential_cfg["shift"],
    )

    integrator_cfg = cfg["integrator"]
    integrator = VelocityVerlet(
        dt=integrator_cfg["dt"],
        ensemble=integrator_cfg["ensemble"],
        thermostat=integrator_cfg["thermostat"],
        target_temperature=integrator_cfg["target_temperature"],
        tau=integrator_cfg["tau"],
        collision_frequency=integrator_cfg["collision_frequency"],
        kb=cfg["system"]["boltzmann_constant"],
    )

    simulation = Simulation(system, potential, integrator, cfg)
    simulation.run()

    output_cfg = cfg["output"]
    output_dir = Path(output_cfg.get("output_dir", "./sim_results"))
    trajectory = np.asarray(simulation.trajectory, dtype=np.float64)

    if not args.no_plots:
        from .visualization import plot_rdf, plot_thermodynamics

        plot_thermodynamics(
            simulation.history,
            save_path=str(output_dir / output_cfg.get("energy_plot", "thermodynamics.png")),
        )
        plot_rdf(
            trajectory,
            system.box,
            save_path=str(output_dir / output_cfg.get("rdf_plot", "rdf.png")),
        )

    if not args.no_animation and len(trajectory) > 0:
        from .visualization import create_3d_animation

        print("Rendering 3D particle animation...")
        create_3d_animation(
            trajectory,
            system.box,
            system.charges,
            system.sigmas,
            save_path=str(output_dir / output_cfg.get("animation", "simulation_3d.gif")),
        )
    return 0
