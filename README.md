# Simple Particle MD

This project implements a vectorized periodic molecular-dynamics simulator with
Lennard-Jones and Coulomb interactions, Velocity-Verlet integration, NVE/NVT
ensembles, thermodynamic logging, RDF analysis, and 3D animation.

## Environment

Create or update the Conda environment described by `environment.yml`:

```bash
conda env create -f environment.yml
# For an existing environment, use: conda env update -n simple_particle_md -f environment.yml
conda activate simple_particle_md
```

The current environment is also named `simple_particle_md`.

## Run

```bash
python main.py configs/ionic_fluid.yaml
```

Use `--no-animation` to skip GIF generation or `--no-plots` to skip the energy
and RDF plots. Results are written to `sim_results/`.
