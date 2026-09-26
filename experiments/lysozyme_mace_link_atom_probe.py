from __future__ import annotations

import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit
from openmmml import MLPotential

PDB_ID = "1LYZ"
MODEL_NAME = os.environ.get("MACE_MODEL", "mace-off24-medium")
TARGETS = (("GLU", "35"), ("ASP", "52"))
STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


def version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def atom_key(atom):
    return atom.residue.name, atom.residue.id, atom.name, atom.index


def qpositions(xyz_nm: np.ndarray, n_virtual: int):
    vecs = [openmm.Vec3(*row) for row in xyz_nm]
    vecs.extend(openmm.Vec3(0.0, 0.0, 0.0) for _ in range(n_virtual))
    return vecs * unit.nanometer


def evaluate(context: openmm.Context):
    context.computeVirtualSites()
    st = context.getState(energy=True, forces=True, positions=True)
    e = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    f = np.asarray(st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
    x = np.asarray(st.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    return float(e), f, x


def group_energy(context: openmm.Context, group: int) -> float:
    return context.getState(energy=True, groups=1 << group).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def select_protein(pdb: app.PDBFile):
    chains = list(pdb.topology.chains())
    scores = []
    for chain in chains:
        residues = list(chain.residues())
        scores.append(sum(r.name in STANDARD_AA for r in residues))
    protein_chain = chains[int(np.argmax(scores))]
    modeller = app.Modeller(pdb.topology, pdb.positions)
    delete_residues = [
        r for r in modeller.topology.residues()
        if r.chain.index != protein_chain.index or r.name not in STANDARD_AA
    ]
    modeller.delete(delete_residues)
    return modeller


def find_residue(topology, name: str, resid: str):
    matches = [r for r in topology.residues() if r.name == name and r.id == resid]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {name}{resid}, found {len(matches)}")
    return matches[0]


def atom_named(residue, name: str):
    matches = [a for a in residue.atoms() if a.name == name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected {residue.name}{residue.id}:{name}, found {len(matches)}")
    return matches[0]


def make_graph(topology):
    graph = {a.index: set() for a in topology.atoms()}
    for b in topology.bonds():
        i, j = b.atom1.index, b.atom2.index
        graph[i].add(j)
        graph[j].add(i)
    return graph


def component_after_cut(topology, ml_atom: int, mm_atom: int):
    graph = make_graph(topology)
    stack = [ml_atom]
    seen = {ml_atom}
    while stack:
        i = stack.pop()
        for j in graph[i]:
            if {i, j} == {ml_atom, mm_atom}:
                continue
            if j not in seen:
                seen.add(j)
                stack.append(j)
    if mm_atom in seen:
        raise RuntimeError("Cut did not disconnect topology")
    return seen


def identify_ml_region(topology):
    ml_atoms = set()
    boundaries = []
    for resname, resid in TARGETS:
        residue = find_residue(topology, resname, resid)
        cb = atom_named(residue, "CB")
        cg = atom_named(residue, "CG")
        component = component_after_cut(topology, cg.index, cb.index)
        # Prevent accidental traversal beyond the intended side chain.
        if any(list(topology.atoms())[i].residue.index != residue.index for i in component):
            raise RuntimeError(f"{resname}{resid} side-chain cut escaped its residue")
        ml_atoms.update(component)
        boundaries.append({
            "label": f"{resname}{resid}",
            "ml": cg.index,
            "mm": cb.index,
            "ml_name": "CG",
            "mm_name": "CB",
        })
    return sorted(ml_atoms), boundaries


def count_disulfides(topology):
    return sum(
        b.atom1.name == "SG" and b.atom2.name == "SG"
        for b in topology.bonds()
    )


def harmonic_bond_pairs(system):
    pairs = set()
    for force in system.getForces():
        if isinstance(force, openmm.HarmonicBondForce):
            for i in range(force.getNumBonds()):
                a, b, _, _ = force.getBondParameters(i)
                pairs.add(tuple(sorted((int(a), int(b)))))
    return pairs


def constraint_pairs(system):
    return {
        tuple(sorted(map(int, system.getConstraintParameters(i)[:2])))
        for i in range(system.getNumConstraints())
    }


def nonbonded_force(system):
    nbs = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)]
    if len(nbs) != 1:
        raise RuntimeError(f"Expected one NonbondedForce, found {len(nbs)}")
    return nbs[0]


def assert_mm_parameters_preserved(mm_system, mixed_system, original_particles):
    a = nonbonded_force(mm_system)
    b = nonbonded_force(mixed_system)
    for i in range(original_particles):
        aq, asig, aeps = a.getParticleParameters(i)
        bq, bsig, beps = b.getParticleParameters(i)
        assert aq == bq and asig == bsig and aeps == beps, i


def assert_ml_nonbonded_excluded(mixed_system, ml_atoms):
    nb = nonbonded_force(mixed_system)
    exceptions = {}
    for i in range(nb.getNumExceptions()):
        a, b, qprod, sigma, epsilon = nb.getExceptionParameters(i)
        exceptions[tuple(sorted((int(a), int(b))))] = (qprod, sigma, epsilon)
    for ia, a in enumerate(ml_atoms):
        for b in ml_atoms[:ia]:
            pair = tuple(sorted((a, b)))
            assert pair in exceptions, f"Missing ML-ML exception {pair}"
            qprod, _, epsilon = exceptions[pair]
            assert abs(qprod.value_in_unit(unit.elementary_charge**2)) < 1e-12, pair
            assert abs(epsilon.value_in_unit(unit.kilojoule_per_mole)) < 1e-12, pair


def assert_bonded_bookkeeping(topology, mm_system, mixed_system, ml_atoms, boundaries):
    ml_set = set(ml_atoms)
    mm_bonds = harmonic_bond_pairs(mm_system)
    mixed_bonds = harmonic_bond_pairs(mixed_system)
    mixed_constraints = constraint_pairs(mixed_system)

    for boundary in boundaries:
        pair = tuple(sorted((boundary["ml"], boundary["mm"])))
        assert pair in mm_bonds, f"Boundary bond absent in MM system: {pair}"
        assert pair in mixed_bonds, f"Boundary bond removed from mixed system: {pair}"

    atoms = list(topology.atoms())
    for b in topology.bonds():
        i, j = b.atom1.index, b.atom2.index
        if i in ml_set and j in ml_set:
            pair = tuple(sorted((i, j)))
            if atoms[i].element.symbol != "H" and atoms[j].element.symbol != "H":
                assert pair not in mixed_bonds, f"Internal ML heavy-atom bond retained: {pair}"
            assert pair not in mixed_constraints, f"Internal ML constraint retained: {pair}"


def match_link_sites(positions, virtual_indices, boundaries):
    assignment = {}
    unused = set(virtual_indices)
    for boundary in boundaries:
        ml = boundary["ml"]
        site = min(unused, key=lambda s: np.linalg.norm(positions[s] - positions[ml]))
        unused.remove(site)
        assignment[boundary["label"]] = site
    return assignment


def assert_link_geometry(positions, boundaries, virtual_indices, tol=2e-7):
    assignment = match_link_sites(positions, virtual_indices, boundaries)
    metrics = {}
    for boundary in boundaries:
        label = boundary["label"]
        ml, mm = boundary["ml"], boundary["mm"]
        site = assignment[label]
        v_mm = positions[mm] - positions[ml]
        v_h = positions[site] - positions[ml]
        d = float(np.linalg.norm(v_h))
        cos = float(np.dot(v_mm, v_h) / (np.linalg.norm(v_mm) * d))
        metrics[label] = {"site": site, "distance_nm": d, "cosine": cos}
        assert abs(d - 0.107) < tol, (label, d)
        assert abs(cos - 1.0) < 1e-9, (label, cos)
    return metrics


def directional_fd(context, xyz_real, n_virtual, atom, direction, delta=1e-5):
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    context.setPositions(qpositions(xyz_real, n_virtual))
    e0, f0, _ = evaluate(context)
    plus = xyz_real.copy(); plus[atom] += delta * direction
    minus = xyz_real.copy(); minus[atom] -= delta * direction
    context.setPositions(qpositions(plus, n_virtual)); ep, _, _ = evaluate(context)
    context.setPositions(qpositions(minus, n_virtual)); em, _, _ = evaluate(context)
    fn = -(ep - em) / (2 * delta)
    fa = float(np.dot(f0[atom], direction))
    abs_err = abs(fa - fn)
    rel_err = abs_err / max(1.0, abs(fa), abs(fn))
    return {"E0": e0, "analytic": fa, "numeric": fn, "abs_error": abs_err, "rel_error": rel_err}


def build_mixed(topology, mm_system, ml_atoms):
    potential = MLPotential(MODEL_NAME)
    return potential.createMixedSystem(
        topology,
        mm_system,
        ml_atoms,
        embedding="mechanical",
        returnInfo=True,
        forceGroup=1,
        precision="double",
        # Glu35 is protonated below and Asp52 is deprotonated: formal ML charge -1.
        # MACE-OFF is not a charge-aware biomolecular model; this argument is included
        # to exercise the current OpenMM-ML/MACE API, not to claim physical accuracy.
        charge=-1,
    )


def run_vacuum(topology, positions, forcefield, ml_atoms, boundaries, result):
    print("\n=== VACUUM WHOLE-PROTEIN TEST ===")
    mm = forcefield.createSystem(topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    info = build_mixed(topology, mm, ml_atoms)
    mixed = info["system"]
    original_n = mm.getNumParticles()
    n_virtual = mixed.getNumParticles() - original_n
    virtual = [i for i in range(mixed.getNumParticles()) if mixed.isVirtualSite(i)]
    assert n_virtual == 2 and virtual == [original_n, original_n + 1]
    assert info["oldToNew"] == list(range(original_n))
    assert info["topology"].getNumAtoms() == mixed.getNumParticles()
    assert_mm_parameters_preserved(mm, mixed, original_n)
    assert_ml_nonbonded_excluded(mixed, ml_atoms)
    assert_bonded_bookkeeping(topology, mm, mixed, ml_atoms, boundaries)

    xyz = np.asarray(positions.value_in_unit(unit.nanometer), dtype=float)
    platform_names = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]
    energies = {}
    force_refs = {}
    cpu_context = None
    cpu_integrator = None
    for pname in ("CPU", "Reference"):
        if pname not in platform_names:
            continue
        integ = openmm.VerletIntegrator(0.5 * unit.femtosecond)
        ctx = openmm.Context(mixed, integ, openmm.Platform.getPlatformByName(pname))
        ctx.setPositions(qpositions(xyz, n_virtual))
        e, f, x = evaluate(ctx)
        g0, g1 = group_energy(ctx, 0), group_energy(ctx, 1)
        assert np.isfinite(e) and np.all(np.isfinite(f))
        assert abs(e - (g0 + g1)) < 1e-5
        geom = assert_link_geometry(x, boundaries, virtual)
        print(f"{pname}: E={e:.9f} kJ/mol  Amber={g0:.9f}  MACE={g1:.9f}  max|F|={np.abs(f).max():.4f}")
        for label, gm in geom.items():
            print(f"  {label}: link={gm['site']} d={gm['distance_nm']:.9f} nm cos={gm['cosine']:.12f}")
        energies[pname] = e
        force_refs[pname] = f
        if pname == "CPU":
            cpu_context, cpu_integrator = ctx, integ
        else:
            del ctx, integ

    assert "CPU" in energies
    if "Reference" in energies:
        assert np.isclose(energies["CPU"], energies["Reference"], rtol=1e-9, atol=1e-4)
        assert np.allclose(force_refs["CPU"], force_refs["Reference"], rtol=1e-7, atol=2e-3)

    fd = {}
    for boundary in boundaries:
        direction = xyz[boundary["mm"]] - xyz[boundary["ml"]]
        for side in ("ml", "mm"):
            atom = boundary[side]
            key = f"{boundary['label']}_{side}"
            check = directional_fd(cpu_context, xyz, n_virtual, atom, direction)
            fd[key] = check
            print(f"FD {key}: analytic={check['analytic']:.7f} numeric={check['numeric']:.7f} rel={check['rel_error']:.3e}")
            assert check["rel_error"] < 2e-3 or check["abs_error"] < 0.5

    # Three independent small boundary perturbations should remain finite.
    rng = np.random.default_rng(20260927)
    pert_atoms = sorted({b["ml"] for b in boundaries} | {b["mm"] for b in boundaries})
    for trial in range(3):
        pert = xyz.copy()
        pert[pert_atoms] += rng.normal(scale=0.001, size=(len(pert_atoms), 3))
        cpu_context.setPositions(qpositions(pert, n_virtual))
        e, f, x = evaluate(cpu_context)
        assert np.isfinite(e) and np.all(np.isfinite(f)) and np.all(np.isfinite(x))
        assert_link_geometry(x, boundaries, virtual)
        print(f"perturbation {trial}: E={e:.9f} max|F|={np.abs(f).max():.4f}")

    result["vacuum"] = {"energies": energies, "finite_difference": fd}


def run_solvated(modeller, forcefield, ml_atoms, boundaries, result):
    print("\n=== EXPLICIT-SOLVENT PERIODIC TEST ===")
    solv = app.Modeller(modeller.topology, modeller.positions)
    solv.addSolvent(
        forcefield,
        model="tip3p",
        padding=0.7 * unit.nanometer,
        ionicStrength=0.05 * unit.molar,
        neutralize=True,
    )
    topology, positions = solv.topology, solv.positions
    # Solvent/ions are appended, so protein atom indices and ML selection are unchanged.
    mm = forcefield.createSystem(
        topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=0.9 * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
    )
    info = build_mixed(topology, mm, ml_atoms)
    mixed = info["system"]
    original_n = mm.getNumParticles()
    n_virtual = mixed.getNumParticles() - original_n
    virtual = [i for i in range(mixed.getNumParticles()) if mixed.isVirtualSite(i)]
    assert n_virtual == 2
    assert_mm_parameters_preserved(mm, mixed, original_n)
    assert_ml_nonbonded_excluded(mixed, ml_atoms)
    assert_bonded_bookkeeping(topology, mm, mixed, ml_atoms, boundaries)

    xyz = np.asarray(positions.value_in_unit(unit.nanometer), dtype=float)
    integ = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.5 * unit.femtosecond)
    ctx = openmm.Context(mixed, integ, openmm.Platform.getPlatformByName("CPU"))
    ctx.setPositions(qpositions(xyz, n_virtual))
    e0, f0, x0 = evaluate(ctx)
    g0, g1 = group_energy(ctx, 0), group_energy(ctx, 1)
    assert np.isfinite(e0) and np.all(np.isfinite(f0))
    assert abs(e0 - (g0 + g1)) < 1e-4
    geom0 = assert_link_geometry(x0, boundaries, virtual)
    print(f"solvated particles={original_n}, virtual={n_virtual}")
    print(f"initial E={e0:.8f} Amber={g0:.8f} MACE={g1:.8f} max|F|={np.abs(f0).max():.4f}")

    # One finite-difference check on each side of the GLU35 boundary under PME.
    b = boundaries[0]
    direction = xyz[b["mm"]] - xyz[b["ml"]]
    periodic_fd = {}
    for side in ("ml", "mm"):
        key = f"{b['label']}_{side}"
        check = directional_fd(ctx, xyz, n_virtual, b[side], direction, delta=2e-5)
        periodic_fd[key] = check
        print(f"PME FD {key}: analytic={check['analytic']:.7f} numeric={check['numeric']:.7f} rel={check['rel_error']:.3e}")
        assert check["rel_error"] < 3e-3 or check["abs_error"] < 1.0

    ctx.setPositions(qpositions(xyz, n_virtual))
    openmm.LocalEnergyMinimizer.minimize(ctx, tolerance=50 * unit.kilojoule_per_mole / unit.nanometer, maxIterations=20)
    emin, fmin, xmin = evaluate(ctx)
    assert np.isfinite(emin) and np.all(np.isfinite(fmin))
    assert emin <= e0 + 1e-4
    assert_link_geometry(xmin, boundaries, virtual)
    print(f"after minimization E={emin:.8f} max|F|={np.abs(fmin).max():.4f}")

    ctx.setVelocitiesToTemperature(300 * unit.kelvin, 20260927)
    integ.step(40)
    emd, fmd, xmd = evaluate(ctx)
    assert np.isfinite(emd) and np.all(np.isfinite(fmd)) and np.all(np.isfinite(xmd))
    geom_md = assert_link_geometry(xmd, boundaries, virtual)
    print(f"after 40 MD steps E={emd:.8f} max|F|={np.abs(fmd).max():.4f}")

    result["solvated"] = {
        "particles": original_n,
        "initial_energy": e0,
        "amber_group_energy": g0,
        "mace_group_energy": g1,
        "minimized_energy": emin,
        "md_energy": emd,
        "finite_difference": periodic_fd,
        "initial_link_geometry": geom0,
        "md_link_geometry": geom_md,
    }


def main():
    print("=== SOFTWARE ===")
    for p in ("openmm", "openmmml", "mace-torch", "torch", "e3nn", "numpy"):
        print(f"{p}: {version(p)}")
    print(f"MACE model: {MODEL_NAME}")
    print(f"OpenMM platforms: {[openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]}")

    pdb_path = Path(os.environ.get("LYSOZYME_PDB", "/tmp/1LYZ.pdb"))
    pdb = app.PDBFile(str(pdb_path))
    modeller = select_protein(pdb)

    forcefield = app.ForceField("amber19-all.xml", "amber19/tip3pfb.xml")
    residues = list(modeller.topology.residues())
    print(f"protein residues before H: {len(residues)}")
    assert 125 <= len(residues) <= 130
    assert count_disulfides(modeller.topology) == 4, f"Expected 4 disulfides, got {count_disulfides(modeller.topology)}"

    # Catalytic-like HEWL protonation: Glu35 protonated, Asp52 deprotonated.
    variants = [None] * len(residues)
    glu35_i = next(i for i, r in enumerate(residues) if r.name == "GLU" and r.id == "35")
    variants[glu35_i] = "GLH"
    modeller.addHydrogens(forcefield, pH=5.2, variants=variants)

    ml_atoms, boundaries = identify_ml_region(modeller.topology)
    atoms = list(modeller.topology.atoms())
    crossing = []
    ml_set = set(ml_atoms)
    for b in modeller.topology.bonds():
        i, j = b.atom1.index, b.atom2.index
        if (i in ml_set) ^ (j in ml_set):
            crossing.append((i, j))
    assert len(crossing) == 2, crossing

    print(f"protein atoms with H: {modeller.topology.getNumAtoms()}")
    print(f"ML atoms: {len(ml_atoms)}")
    print("ML atom identities:")
    for i in ml_atoms:
        print(" ", atom_key(atoms[i]))
    print("boundaries:")
    for b in boundaries:
        print(f"  {b['label']}: {atom_key(atoms[b['mm']])} -- {atom_key(atoms[b['ml']])}")

    result = {
        "pdb": PDB_ID,
        "model": MODEL_NAME,
        "versions": {p: version(p) for p in ("openmm", "openmmml", "mace-torch", "torch", "e3nn", "numpy")},
        "protein_residues": len(list(modeller.topology.residues())),
        "protein_atoms": modeller.topology.getNumAtoms(),
        "disulfides": count_disulfides(modeller.topology),
        "ml_atom_count": len(ml_atoms),
        "boundaries": boundaries,
    }

    run_vacuum(modeller.topology, modeller.positions, forcefield, ml_atoms, boundaries, result)
    run_solvated(modeller, forcefield, ml_atoms, boundaries, result)

    Path("lysozyme-mace-link-atom-results.json").write_text(json.dumps(result, indent=2))
    print("\nLYSOZYME_MACE_LINK_ATOM_PROBE_PASS")


if __name__ == "__main__":
    main()
