# Verified HEWL MACE/OpenMM-ML link-atom compatibility test

Verified CI run: `36277514076`, tested code/workflow commit `ade841bb05c193d6b11b4217fa04af9994a1e4ec`.

## Software

- Python 3.11.16
- OpenMM 8.6.1
- OpenMM-ML 1.8, commit `a7fb40ebecd031db3a5d1da08202cad6819c366f`
- mace-torch 0.3.16
- PyTorch 2.14.0+cpu
- e3nn 0.4.4
- NumPy 2.4.6
- MACE model: `mace-off24-medium`

## Protein and partition

- Hen egg-white lysozyme, PDB 1LYZ
- 129 protein residues and all 4 native disulfide bonds retained
- Glu35 explicitly protonated; Asp52 deprotonated
- 1962 protein atoms after adding hydrogens
- Amber19 protein force field
- ML region: 10 atoms, comprising the distal side-chain fragments of Glu35 and Asp52
- Glu35 ML/MM boundary: CB(540)--CG(543)
- Asp52 ML/MM boundary: CB(780)--CG(783)
- Two hydrogen link atoms were appended as virtual sites

The ML atoms were Glu35 CG, HG2, HG3, CD, OE1, OE2, HE2 and Asp52 CG, OD1, OD2.

## Structural/bookkeeping checks

The test asserted all of the following before evaluating dynamics:

- exactly two virtual link H atoms are appended;
- the original-to-new atom-index mapping is unchanged for all real particles;
- the modified topology and system particle counts agree;
- original Amber per-particle nonbonded parameters are preserved;
- all ML--ML Amber nonbonded interactions are excluded;
- the two CB--CG boundary bonds remain in the MM bonded description;
- internal ML bonded terms/constraints that OpenMM-ML is meant to replace are removed;
- both link H atoms remain at 0.107000000 nm from the ML-side CG atoms and exactly collinear with the CB--CG boundary axes.

## Vacuum whole-protein test

Amber-prepared coordinates were used before constructing the mixed system.

CPU result:

- total energy: -15761.959168729 kJ/mol
- Amber force-group energy: -10797.341623231 kJ/mol
- MACE force-group energy: -4964.617545498 kJ/mol
- maximum absolute force component: 6010.7279 kJ/mol/nm

Reference-platform result:

- total energy: -15761.963584530 kJ/mol
- Amber force-group energy: -10797.346039032 kJ/mol
- MACE force-group energy: -4964.617545498 kJ/mol

The MACE force-group energy was identical to printed precision between CPU and Reference platforms.

### Directional finite-difference checks

The strict derivative test finite-differenced the MACE force group itself, which directly tests the MACE `PythonForce` plus OpenMM virtual-site force redistribution without cancellation against the much larger Amber energy.

| Boundary atom | Analytic MACE force (kJ/mol/nm) | Numerical MACE force | Relative error |
|---|---:|---:|---:|
| Glu35 CG (ML side) | -581.5789113 | -581.6918119 | 1.941e-4 |
| Glu35 CB (MM side) | 0.0000000 | -0.0000000 | 6.750e-14 |
| Asp52 CG (ML side) | -2224.0448214 | -2223.9660639 | 3.541e-5 |
| Asp52 CB (MM side) | -0.0000000 | -0.0000000 | 2.771e-13 |

The whole-system finite differences were retained as independent diagnostics. Their relative errors were 7.033e-4, 6.490e-3, 1.113e-3 and 7.469e-3 for Glu35-CG, Glu35-CB, Asp52-CG and Asp52-CB respectively. The larger relative errors on the MM-side atoms reflect cancellation between large force components; the force-group-specific MACE derivatives are the decisive virtual-site consistency test.

Three independent random perturbations of the four boundary atoms all produced finite energies/forces and preserved exact link-site geometry.

## Explicit-solvent periodic PME test

The protein was solvated with TIP3P water using 0.7 nm padding, neutralized, and set to 0.05 M ionic strength. The periodic system contained 23245 real particles plus two link virtual sites. Amber used PME with a 0.9 nm cutoff.

Initial mixed-system result:

- total energy: -337818.94589187 kJ/mol
- Amber force-group energy: -332854.16849046 kJ/mol
- MACE force-group energy: -4964.77740142 kJ/mol
- maximum absolute force component: 21835.7111 kJ/mol/nm

The force decomposition contained only groups 0 (Amber) and 1 (MACE `PythonForce`), and the separately queried group energies summed to the total with a residual of exactly 0 kJ/mol in this run.

Glu35 PME boundary derivative checks:

| Boundary atom | Analytic MACE force (kJ/mol/nm) | Numerical MACE force | Relative error |
|---|---:|---:|---:|
| Glu35 CG (ML side) | -282.4720929 | -282.5918191 | 4.237e-4 |
| Glu35 CB (MM side) | 0.0000000 | -0.0000000 | 1.954e-14 |

Whole-system PME finite-difference relative errors were 2.766e-3 (CG) and 8.752e-4 (CB).

Mixed-system minimization lowered the energy to -368792.42576263 kJ/mol. A subsequent 40-step 300 K Langevin trajectory completed with finite energy and forces; final energy was -357264.70782301 kJ/mol. Both link atoms retained the prescribed 0.107 nm, collinear geometry throughout.

The CI terminated with `LYSOZYME_MACE_LINK_ATOM_PROBE_PASS` and every workflow step completed successfully.

## Interpretation and limitations

This establishes software-level compatibility of MACE 0.3.16, OpenMM 8.6.1 and current OpenMM-ML link atoms for a genuine protein topology containing two covalent ML/MM boundaries, including periodic explicit solvent and PME on the MM side.

It does **not** establish that MACE-OFF24-medium is an appropriate physical potential for enzyme active-site chemistry. MACE-OFF24 is being used here as a convenient compatible MACE model to stress the coupling machinery. The chosen ML subsystem is also deliberately tiny and lacks the carbohydrate substrate normally included in mechanistic HEWL QM/MM calculations. A chemically predictive calculation would require an ML potential trained/validated for the relevant charged/protonation states and reactive chemical space, plus a scientifically justified ML-region definition.