"""Configuration loading, defaulting, and schema validation."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Union

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only without the optional dependency
    yaml = None


PathLike = Union[str, os.PathLike]


def _require_mapping(value: Any, section: str) -> Dict[str, Any]:
    """Return a mutable configuration section or raise a useful error."""
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration section '{section}' must be a mapping.")
    return dict(value)


def _as_positive_float(value: Any, name: str, *, allow_zero: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{name}' must be a number.") from exc

    if not math.isfinite(number) or (number < 0 if allow_zero else number <= 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"'{name}' must be {qualifier}.")
    return number


def load_config(config_path: PathLike) -> Dict[str, Any]:
    """Load a YAML configuration, apply defaults, and validate its values."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Configuration path is not a file: {path}")

    if yaml is None:
        raise ImportError(
            "PyYAML is required to parse YAML files. Install it via `pip install pyyaml`."
        )

    with path.open("r", encoding="utf-8") as config_file:
        cfg = yaml.safe_load(config_file)

    if cfg is None:
        cfg = {}
    if not isinstance(cfg, Mapping):
        raise ValueError("The top-level YAML value must be a mapping.")
    cfg = dict(cfg)

    section_names = ("system", "potential", "integrator", "simulation", "output")
    for section in section_names:
        cfg[section] = _require_mapping(cfg.get(section, {}), section)

    # System defaults
    system = cfg["system"]
    system.setdefault("box_size", [10.0, 10.0, 10.0])
    system.setdefault("initial_temperature", 1.0)
    system.setdefault("boltzmann_constant", 1.0)
    system.setdefault("lattice", "grid")

    # Potential defaults
    potential = cfg["potential"]
    potential.setdefault("cutoff_radius", 4.0)
    potential.setdefault("coulomb_constant", 1.0)
    potential.setdefault("shift", True)

    # Integrator defaults
    integrator = cfg["integrator"]
    integrator.setdefault("type", "velocity_verlet")
    integrator.setdefault("dt", 0.002)
    integrator.setdefault("ensemble", "NVT")
    integrator.setdefault("thermostat", "berendsen")
    integrator.setdefault("target_temperature", 1.0)
    integrator.setdefault("tau", 0.05)
    integrator.setdefault("collision_frequency", 1.0)

    # Simulation defaults
    simulation = cfg["simulation"]
    simulation.setdefault("total_steps", 1000)
    simulation.setdefault("log_interval", 20)
    simulation.setdefault("trajectory_interval", 10)

    # Output defaults
    output = cfg["output"]
    output.setdefault("output_dir", "./sim_results")
    output.setdefault("log_file", "thermodynamics.csv")
    output.setdefault("trajectory_file", "trajectory.npz")
    output.setdefault("energy_plot", "thermodynamics.png")
    output.setdefault("rdf_plot", "rdf.png")
    output.setdefault("animation", "simulation_3d.gif")

    _validate_config(cfg)
    return cfg


def _validate_config(cfg: Mapping[str, Any]) -> None:
    """Validate values that would otherwise produce confusing numerical errors."""
    system = cfg["system"]
    box = system["box_size"]
    if not isinstance(box, (list, tuple)) or len(box) != 3:
        raise ValueError("'system.box_size' must contain exactly three values.")
    box_values = [_as_positive_float(value, "system.box_size", allow_zero=False) for value in box]

    _as_positive_float(system["initial_temperature"], "system.initial_temperature", allow_zero=True)
    _as_positive_float(system["boltzmann_constant"], "system.boltzmann_constant")
    lattice = str(system["lattice"]).lower()
    if lattice not in {"grid", "random"}:
        raise ValueError("'system.lattice' must be either 'grid' or 'random'.")

    species = system.get("species", [])
    if not isinstance(species, list) or not species:
        raise ValueError("'system.species' must be a non-empty list.")
    for index, item in enumerate(species):
        if not isinstance(item, Mapping):
            raise ValueError(f"system.species[{index}] must be a mapping.")
        count = item.get("count", 1)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"system.species[{index}].count must be a positive integer.")
        _as_positive_float(item.get("mass", 1.0), f"system.species[{index}].mass")
        _as_positive_float(item.get("sigma", 1.0), f"system.species[{index}].sigma")
        _as_positive_float(
            item.get("epsilon", 1.0), f"system.species[{index}].epsilon", allow_zero=True
        )
        charge = float(item.get("charge", 0.0))
        if not math.isfinite(charge):
            raise ValueError(f"system.species[{index}].charge must be finite.")

    cutoff = _as_positive_float(cfg["potential"]["cutoff_radius"], "potential.cutoff_radius")
    if cutoff > min(box_values) / 2.0:
        raise ValueError("'potential.cutoff_radius' must be <= half the smallest box dimension.")
    _as_positive_float(
        cfg["potential"]["coulomb_constant"], "potential.coulomb_constant", allow_zero=True
    )

    integrator = cfg["integrator"]
    if str(integrator["type"]).lower() != "velocity_verlet":
        raise ValueError("Only 'velocity_verlet' is currently supported.")
    ensemble = str(integrator["ensemble"]).upper()
    if ensemble not in {"NVE", "NVT"}:
        raise ValueError("'integrator.ensemble' must be either 'NVE' or 'NVT'.")
    thermostat = str(integrator["thermostat"]).lower()
    if thermostat not in {"berendsen", "andersen"}:
        raise ValueError("'integrator.thermostat' must be 'berendsen' or 'andersen'.")
    _as_positive_float(integrator["dt"], "integrator.dt")
    _as_positive_float(
        integrator["target_temperature"], "integrator.target_temperature", allow_zero=True
    )
    _as_positive_float(integrator["tau"], "integrator.tau")
    _as_positive_float(
        integrator["collision_frequency"], "integrator.collision_frequency", allow_zero=True
    )

    simulation = cfg["simulation"]
    for key in ("total_steps", "log_interval", "trajectory_interval"):
        value = simulation[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"'simulation.{key}' must be an integer.")
        minimum = 0 if key == "total_steps" else 1
        if value < minimum:
            qualifier = "non-negative" if minimum == 0 else "positive"
            raise ValueError(f"'simulation.{key}' must be {qualifier}.")

    output = cfg["output"]
    for key in ("output_dir", "log_file", "trajectory_file", "energy_plot", "rdf_plot", "animation"):
        value = output[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"'output.{key}' must be a non-empty string.")
