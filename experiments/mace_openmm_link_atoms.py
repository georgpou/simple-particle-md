from __future__ import annotations

import importlib.metadata as metadata
import os
from pathlib import Path

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit

from openmmml import MLPotential


def pkg_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def vec_positions(xyz_nm: np.ndarray, n_virtual: int):
    vectors = [openmm.Vec3(*row) for row in xyz_nm]
    vectors.extend(openmm.Vec3(0.0, 0.0, 0.0) for _ in range(n_virtual))
    return vectors * unit.nanometer


def evaluate(context: openmm.Context):
    context.computeVirtualSites()
    state = context.getState(energy=True, forces=True, positions=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    return energy, np.asarray(forces), np.asarray(positions)


def main() -> None:
    print("=== package versions ===")
    for package in ("openmm", "openmmml", "mace-torch", "torch", "e3nn", "numpy"):
        print(f"{package}: {pkg_version(package)}")

    print("=== OpenMM platforms ===")
    platform_names = [
        openmm.Platform.getPlatform(i).getName()
        for i in range(openmm.Platform.getNumPlatforms())
    ]
    print(platform_names)

    source_root = Path(os.environ["OPENMM_ML_ROOT"]).resolve()
    data_dir = source_root / "test" / "data" / "ethanol"
    pdb_path = data_dir / "ethanol.pdb"
    ff_path = data_dir / "ethanol.xml"
    assert pdb_path.is_file(), pdb_path
    assert ff_path.is_file(), ff_path

    pdb = app.PDBFile(str(pdb_path))
    forcefield = app.ForceField(str(ff_path))
    mm_system = forcefield.createSystem(pdb.topology)

    # Upstream OpenMM-ML link-atom test partition:
    #
    #          H4   H6
    #          |    |
    # H3 - O0 - C1 - C2 - H8
    #          |    |
    #          H5   H7
    #
    # The ML region is O0/C1/H3/H4/H5, so C1-C2 is cut and capped by H.
    ml_atoms = [0, 1, 3, 4, 5]
    original_particles = mm_system.getNumParticles()

    print("=== constructing MACE mixed system ===")
    potential = MLPotential("mace-off23-small")
    info = potential.createMixedSystem(
        pdb.topology,
        mm_system,
        ml_atoms,
        embedding="mechanical",
        returnInfo=True,
        precision="double",
    )
    mixed_system = info["system"]
    mixed_topology = info["topology"]

    n_virtual = mixed_system.getNumParticles() - original_particles
    virtual_indices = [
        i for i in range(mixed_system.getNumParticles()) if mixed_system.isVirtualSite(i)
    ]

    print(f"original particles: {original_particles}")
    print(f"mixed particles:    {mixed_system.getNumParticles()}")
    print(f"virtual sites:      {virtual_indices}")
    print(f"oldToNew:           {info['oldToNew']}")

    assert n_virtual == 1, f"Expected one link atom, found {n_virtual}"
    assert virtual_indices == [original_particles]
    assert mixed_topology.getNumAtoms() == mixed_system.getNumParticles()
    assert info["oldToNew"] == list(range(original_particles))

    xyz0 = np.asarray(pdb.positions.value_in_unit(unit.nanometer), dtype=float)
    full_positions = vec_positions(xyz0, n_virtual)

    tested_platforms = [name for name in ("CPU", "Reference") if name in platform_names]
    assert tested_platforms, "Neither CPU nor Reference OpenMM platform is available"

    base_force = None
    base_energy = None
    cpu_context = None
    cpu_integrator = None

    print("=== evaluating energy/forces and virtual-site geometry ===")
    for platform_name in tested_platforms:
        integrator = openmm.LangevinMiddleIntegrator(
            300.0 * unit.kelvin,
            1.0 / unit.picosecond,
            0.5 * unit.femtosecond,
        )
        platform = openmm.Platform.getPlatformByName(platform_name)
        context = openmm.Context(mixed_system, integrator, platform)
        context.setPositions(full_positions)
        energy, forces, positions = evaluate(context)

        assert np.isfinite(energy), f"Non-finite energy on {platform_name}: {energy}"
        assert np.all(np.isfinite(forces)), f"Non-finite force on {platform_name}"

        c1, c2, link = 1, 2, virtual_indices[0]
        d_cc = positions[c2] - positions[c1]
        d_ch = positions[link] - positions[c1]
        cc = np.linalg.norm(d_cc)
        ch = np.linalg.norm(d_ch)
        cos_angle = float(np.dot(d_cc, d_ch) / (cc * ch))

        print(
            f"{platform_name}: E={energy:.10f} kJ/mol, "
            f"|F|max={np.abs(forces).max():.6f} kJ/mol/nm, "
            f"C1-C2={cc:.8f} nm, C1-Hlink={ch:.8f} nm, cos={cos_angle:.12f}"
        )

        assert np.isclose(ch, 0.107, atol=1e-7), ch
        assert np.isclose(cos_angle, 1.0, atol=1e-10), cos_angle

        if base_energy is None:
            base_energy = energy
            base_force = forces
        else:
            # PythonForce should produce effectively the same result on CPU and Reference.
            assert np.isclose(energy, base_energy, rtol=1e-8, atol=1e-5), (
                platform_name,
                energy,
                base_energy,
            )
            assert np.allclose(forces, base_force, rtol=1e-7, atol=1e-4), platform_name

        if platform_name == "CPU":
            cpu_context = context
            cpu_integrator = integrator
        else:
            del context
            del integrator

    if cpu_context is None:
        platform = openmm.Platform.getPlatformByName(tested_platforms[0])
        cpu_integrator = openmm.LangevinMiddleIntegrator(
            300.0 * unit.kelvin,
            1.0 / unit.picosecond,
            0.5 * unit.femtosecond,
        )
        cpu_context = openmm.Context(mixed_system, cpu_integrator, platform)
        cpu_context.setPositions(full_positions)

    print("=== finite-difference force check across link boundary ===")
    # Check the x force on the ML-side boundary carbon.  Moving this atom also
    # moves the virtual link H, so this checks OpenMM's redistribution of the
    # PythonForce contribution through the LocalCoordinatesSite.
    atom = 1
    axis = 0
    delta_nm = 1.0e-5

    cpu_context.setPositions(vec_positions(xyz0, n_virtual))
    e0, f0, _ = evaluate(cpu_context)

    plus = xyz0.copy()
    minus = xyz0.copy()
    plus[atom, axis] += delta_nm
    minus[atom, axis] -= delta_nm

    cpu_context.setPositions(vec_positions(plus, n_virtual))
    e_plus, _, _ = evaluate(cpu_context)
    cpu_context.setPositions(vec_positions(minus, n_virtual))
    e_minus, _, _ = evaluate(cpu_context)

    numerical_force = -(e_plus - e_minus) / (2.0 * delta_nm)
    analytic_force = f0[atom, axis]
    abs_error = abs(numerical_force - analytic_force)
    rel_error = abs_error / max(1.0, abs(numerical_force), abs(analytic_force))

    print(f"E0:                {e0:.10f} kJ/mol")
    print(f"analytic F(C1,x):  {analytic_force:.8f} kJ/mol/nm")
    print(f"numeric  F(C1,x):  {numerical_force:.8f} kJ/mol/nm")
    print(f"absolute error:    {abs_error:.8f} kJ/mol/nm")
    print(f"relative error:    {rel_error:.3e}")

    assert rel_error < 5.0e-3 or abs_error < 0.5, (
        analytic_force,
        numerical_force,
        abs_error,
        rel_error,
    )

    print("=== minimization and short MD ===")
    cpu_context.setPositions(vec_positions(xyz0, n_virtual))
    e_before, _, _ = evaluate(cpu_context)
    openmm.LocalEnergyMinimizer.minimize(cpu_context, maxIterations=25)
    e_min, f_min, pos_min = evaluate(cpu_context)

    print(f"energy before minimization: {e_before:.10f} kJ/mol")
    print(f"energy after minimization:  {e_min:.10f} kJ/mol")
    print(f"max force after minimize:   {np.abs(f_min).max():.6f} kJ/mol/nm")
    assert np.isfinite(e_min)
    assert np.all(np.isfinite(f_min))
    assert e_min <= e_before + 1e-5

    cpu_context.setVelocitiesToTemperature(300.0 * unit.kelvin, 20260927)
    cpu_integrator.step(20)
    e_md, f_md, pos_md = evaluate(cpu_context)
    assert np.isfinite(e_md)
    assert np.all(np.isfinite(f_md))
    assert np.all(np.isfinite(pos_md))

    c1, c2, link = 1, 2, virtual_indices[0]
    d_cc = pos_md[c2] - pos_md[c1]
    d_ch = pos_md[link] - pos_md[c1]
    ch_md = np.linalg.norm(d_ch)
    cos_md = float(np.dot(d_cc, d_ch) / (np.linalg.norm(d_cc) * ch_md))
    print(
        f"after 20 steps: E={e_md:.10f} kJ/mol, "
        f"C1-Hlink={ch_md:.8f} nm, cos={cos_md:.12f}"
    )
    assert np.isclose(ch_md, 0.107, atol=1e-7)
    assert np.isclose(cos_md, 1.0, atol=1e-10)

    print("COMPATIBILITY_PROBE_PASS")


if __name__ == "__main__":
    main()
