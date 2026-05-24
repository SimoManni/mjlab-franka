"""Map the Franka end-effector reachable workspace via FK random sampling.

Samples ~1M random joint configurations within the soft joint limits,
computes forward kinematics to get the EE (attachment_site) position for each,
then fits a convex hull to the resulting point cloud.

Output:
    src/mjlab_franka/robots/franka/ee_workspace.npz
        - vertices:  (V, 3) float64 — convex hull vertex positions
        - equations: (F, 4) float64 — half-plane equations (Ax + b <= 0)

Usage:
    uv run python scripts/map_ee_workspace.py [--num-samples N] [--plot]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mjlab_franka.robots.franka.franka_constants import (  # noqa: E402
    ARM_JOINT_NAMES,
    EE_SITE_NAME,
    get_spec,
)

OUTPUT_PATH = (
    REPO_ROOT / "src" / "mjlab_franka" / "robots" / "franka" / "ee_workspace.npz"
)

# Matches the soft_joint_pos_limit_factor used in the articulation config.
SOFT_LIMIT_FACTOR = 0.9


def _get_joint_limits(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """Return (lower, upper) soft joint limits for the arm joints (7,)."""
    lower = np.zeros(len(ARM_JOINT_NAMES))
    upper = np.zeros(len(ARM_JOINT_NAMES))
    for i, name in enumerate(ARM_JOINT_NAMES):
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo, hi = model.jnt_range[jnt_id]
        # Apply soft limit factor: shrink range symmetrically.
        mid = (lo + hi) / 2.0
        half = (hi - lo) / 2.0 * SOFT_LIMIT_FACTOR
        lower[i] = mid - half
        upper[i] = mid + half
    return lower, upper


def _get_site_id(model: mujoco.MjModel, site_name: str) -> int:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"Site '{site_name}' not found in model")
    return site_id


def sample_workspace(
    num_samples: int = 1_000_000,
    seed: int = 42,
) -> np.ndarray:
    """Sample EE positions by randomizing joint configs and running FK.

    Returns (num_samples, 3) array of EE positions in world frame.
    """
    spec = get_spec()
    model = spec.compile()
    data = mujoco.MjData(model)

    lower, upper = _get_joint_limits(model)
    site_id = _get_site_id(model, EE_SITE_NAME)

    # Find the qpos indices for arm joints.
    qpos_indices = []
    for name in ARM_JOINT_NAMES:
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_adr = model.jnt_qposadr[jnt_id]
        qpos_indices.append(qpos_adr)
    qpos_indices = np.array(qpos_indices)

    rng = np.random.default_rng(seed)
    ee_positions = np.empty((num_samples, 3), dtype=np.float64)

    print(f"Sampling {num_samples:,} joint configurations...")
    t0 = time.perf_counter()

    # Batch random configs for efficiency.
    all_configs = rng.uniform(lower, upper, size=(num_samples, len(ARM_JOINT_NAMES)))

    for i in range(num_samples):
        data.qpos[qpos_indices] = all_configs[i]
        mujoco.mj_kinematics(model, data)
        ee_positions[i] = data.site_xpos[site_id].copy()

    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s ({num_samples / elapsed:.0f} samples/s)")
    return ee_positions


def compute_and_save_hull(ee_positions: np.ndarray, output_path: Path) -> ConvexHull:
    """Compute convex hull and save to .npz."""
    print(f"Computing convex hull from {len(ee_positions):,} points...")
    hull = ConvexHull(ee_positions)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        vertices=hull.points[hull.vertices],
        equations=hull.equations,
    )
    print(f"  Saved to {output_path}")
    print(f"  Hull vertices: {len(hull.vertices)}")
    print(f"  Hull faces:    {len(hull.equations)}")
    print(f"  Hull volume:   {hull.volume:.4f} m³")

    # Axis-aligned bounds of the hull.
    verts = hull.points[hull.vertices]
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    print(f"  Bounds X: [{lo[0]:.3f}, {hi[0]:.3f}] m")
    print(f"  Bounds Y: [{lo[1]:.3f}, {hi[1]:.3f}] m")
    print(f"  Bounds Z: [{lo[2]:.3f}, {hi[2]:.3f}] m")

    return hull


def plot_workspace(ee_positions: np.ndarray, hull: ConvexHull) -> None:
    """Show a 3D scatter of sampled EE positions + convex hull."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Subsample points for plotting.
    n_plot = min(10_000, len(ee_positions))
    idx = np.random.default_rng(0).choice(len(ee_positions), n_plot, replace=False)
    ax.scatter(
        ee_positions[idx, 0],
        ee_positions[idx, 1],
        ee_positions[idx, 2],
        s=0.5,
        alpha=0.3,
        c="blue",
    )

    # Draw hull faces.
    faces = []
    for simplex in hull.simplices:
        face = hull.points[simplex]
        faces.append(face)
    poly = Poly3DCollection(
        faces, alpha=0.15, facecolor="cyan", edgecolor="gray", linewidths=0.2
    )
    ax.add_collection3d(poly)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Franka EE Reachable Workspace (Convex Hull)")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Map Franka EE workspace via FK sampling"
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=1_000_000,
        help="Number of random joint configurations to sample (default: 1M)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show a 3D matplotlib plot of the workspace",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output .npz path (default: {OUTPUT_PATH.relative_to(REPO_ROOT)})",
    )
    args = parser.parse_args()

    ee_positions = sample_workspace(num_samples=args.num_samples)
    hull = compute_and_save_hull(ee_positions, args.output)

    if args.plot:
        plot_workspace(ee_positions, hull)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
