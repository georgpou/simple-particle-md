"""Thermodynamic plots, radial distribution functions, and 3D animations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def _prepare_output_path(save_path: str) -> Path:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_thermodynamics(
    history: List[Dict[str, Any]], save_path: Optional[str] = None
) -> None:
    """Plot potential, kinetic, total energy, and temperature over time."""
    if not history:
        raise ValueError("Cannot plot thermodynamics without history data.")

    time = [entry["time"] for entry in history]
    e_tot = [entry["e_tot"] for entry in history]
    e_pot = [entry["e_pot"] for entry in history]
    e_kin = [entry["e_kin"] for entry in history]
    temperature = [entry["temperature"] for entry in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax1.plot(time, e_tot, label="Total Energy", color="#222222", lw=2)
    ax1.plot(time, e_pot, label="Potential Energy", color="#d9534f", lw=1.5)
    ax1.plot(time, e_kin, label="Kinetic Energy", color="#0275d8", lw=1.5)
    ax1.set_ylabel("Energy [units]")
    ax1.set_title("Thermodynamic Trajectory")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(time, temperature, label="Instantaneous Temp", color="#f0ad4e", lw=1.5)
    ax2.set_xlabel("Time [units]")
    ax2.set_ylabel("Temperature")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    if save_path:
        path = _prepare_output_path(save_path)
        fig.savefig(path, dpi=200)
        print(f"[+] Energy plot saved: {path}")
    plt.close(fig)


def compute_rdf(
    trajectory: np.ndarray,
    box: np.ndarray,
    r_max: Optional[float] = None,
    n_bins: int = 80,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the all-particle radial distribution function ``g(r)``."""
    trajectory = np.asarray(trajectory, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)
    if trajectory.ndim != 3 or trajectory.shape[2:] != (3,):
        raise ValueError("trajectory must have shape (n_frames, n_particles, 3).")
    if trajectory.shape[0] == 0:
        raise ValueError("Cannot compute an RDF from an empty trajectory.")
    if box.shape != (3,) or np.any(box <= 0):
        raise ValueError("box must contain three positive dimensions.")
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer.")

    n_frames, n_particles, _ = trajectory.shape
    if r_max is None:
        r_max = float(np.min(box) / 2.0)
    r_max = float(r_max)
    if not np.isfinite(r_max) or r_max <= 0 or r_max > np.min(box) / 2.0:
        raise ValueError("r_max must be positive and no greater than half the smallest box side.")

    edges = np.linspace(0.0, r_max, n_bins + 1)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    histogram = np.zeros(n_bins, dtype=np.float64)
    upper_indices = np.triu_indices(n_particles, k=1)

    for frame in trajectory:
        displacement = frame[np.newaxis, :, :] - frame[:, np.newaxis, :]
        displacement -= box * np.round(displacement / box)
        distances = np.sqrt(np.sum(displacement**2, axis=-1))
        counts, _ = np.histogram(distances[upper_indices], bins=edges)
        histogram += counts

    histogram /= n_frames
    density = n_particles / float(np.prod(box))
    shell_volumes = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    ideal_counts = 0.5 * n_particles * density * shell_volumes
    gr = np.divide(histogram, ideal_counts, out=np.zeros_like(histogram), where=ideal_counts > 0)
    return bin_centers, gr


def plot_rdf(
    trajectory: np.ndarray,
    box: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Compute and save the radial distribution function curve."""
    r_values, gr = compute_rdf(trajectory, box)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(r_values, gr, color="#5cb85c", lw=2.0)
    ax.axhline(1.0, color="gray", linestyle=":", lw=1)
    ax.set_xlabel("Distance r [units]")
    ax.set_ylabel("g(r)")
    ax.set_title("Radial Distribution Function")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    if save_path:
        path = _prepare_output_path(save_path)
        fig.savefig(path, dpi=200)
        print(f"[+] RDF plot saved: {path}")
    plt.close(fig)


def create_3d_animation(
    trajectory: np.ndarray,
    box: np.ndarray,
    charges: np.ndarray,
    sigmas: np.ndarray,
    save_path: str,
    fps: int = 20,
) -> None:
    """Render and save a 3D particle animation as GIF or MP4."""
    trajectory = np.asarray(trajectory, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)
    charges = np.asarray(charges, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    if trajectory.ndim != 3 or trajectory.shape[0] == 0 or trajectory.shape[2:] != (3,):
        raise ValueError("trajectory must contain at least one frame with shape (N, 3).")
    if box.shape != (3,) or np.any(box <= 0):
        raise ValueError("box must contain three positive dimensions.")
    if charges.shape != (trajectory.shape[1],) or sigmas.shape != charges.shape:
        raise ValueError("charges and sigmas must contain one value per particle.")
    if np.any(sigmas <= 0):
        raise ValueError("sigmas must be positive.")
    if isinstance(fps, bool) or not isinstance(fps, int) or fps < 1:
        raise ValueError("fps must be a positive integer.")

    output_path = _prepare_output_path(save_path)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    lx, ly, lz = box

    for x in (0, lx):
        for y in (0, ly):
            ax.plot([x, x], [y, y], [0, lz], color="gray", linestyle="--", lw=0.7)
    for x in (0, lx):
        for z in (0, lz):
            ax.plot([x, x], [0, ly], [z, z], color="gray", linestyle="--", lw=0.7)
    for y in (0, ly):
        for z in (0, lz):
            ax.plot([0, lx], [y, y], [z, z], color="gray", linestyle="--", lw=0.7)

    ax.set_xlim(0, lx)
    ax.set_ylim(0, ly)
    ax.set_zlim(0, lz)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    colors = [
        "#e74c3c" if charge > 0.05 else "#3498db" if charge < -0.05 else "#95a5a6"
        for charge in charges
    ]
    sizes = (sigmas / np.mean(sigmas)) ** 2 * 90.0
    first_frame = trajectory[0]
    scatter = ax.scatter(
        first_frame[:, 0],
        first_frame[:, 1],
        first_frame[:, 2],
        c=colors,
        s=sizes,
        edgecolors="black",
        alpha=0.85,
    )
    title = ax.set_title("Simulation Frame: 0")

    def update(frame: int):
        coords = trajectory[frame]
        scatter._offsets3d = (coords[:, 0], coords[:, 1], coords[:, 2])
        title.set_text(f"Simulation Frame: {frame} / {len(trajectory) - 1}")
        return scatter, title

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(trajectory),
        interval=max(1, int(1000 / fps)),
        blit=False,
    )
    writer = animation.FFMpegWriter(fps=fps) if output_path.suffix.lower() == ".mp4" else animation.PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close(fig)
    print(f"[+] 3D animation saved: {output_path}")
