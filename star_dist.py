"""
star_dist.py

Main executable for creating and displaying collections of star clusters
"""

import os
import argparse
from typing import Dict, Any, Tuple
import yaml
import numpy as np
import matplotlib.pyplot as plt

import cluster_simulation as cs
import cluster_common as cc


def settings() -> argparse.Namespace:
    """
    Process the CLI arguments and return the corresponding arguments

    Return:
        Arguments passed from the CLI invocation
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        help="YAML file containing cluster simulation configuration",
        required=True,
    )
    return parser.parse_args()


def compose(config: Dict[str, Any]) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Given a simulation configuration, return a tuple containing the
    mass, position and velocities as a set of numpy arrays.

    Args:
        config - Dictionary containing the simulation configuration

    Return:
        Tuple containing the mass, position and velocity vector defining the
        cluster
    """
    sim = cs.Simulation(config=config[cc.SIMULATION])

    mass = []
    pos = []
    vel = []

    for m, p, v in sim.gen_clusters():
        mass.append(m)
        pos.append(p)
        vel.append(v)

    return np.concatenate(mass), np.concatenate(pos), np.concatenate(vel)


def plot(name: str, pos: np.ndarray, vel: np.ndarray, subsample: int = 1000):
    """
    Plot the position and velocity of a collection of stars.

    Args:
        name - Name of cluster to display in the plot title
        pos - Numpy array of positions of the stars to plot
        vel - Numpy array of velocities of stars to plot
        subsample - Number of elements to display
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot positions
    ax.scatter(
        pos[:, 0],
        pos[:, 1],
        zs=pos[:, 2],
        s=0.5,
        alpha=0.6,
        color="blue",
    )

    # Plot velocity vectors (sub-sampled)
    idx = np.random.choice(
        len(pos),
        size=min(subsample, len(pos)),
        replace=False,
    )
    ax.quiver(
        pos[idx, 0],
        pos[idx, 1],
        pos[idx, 2],
        vel[idx, 0],
        vel[idx, 1],
        vel[idx, 2],
        length=0.5,
        normalize=True,
        color="red",
        alpha=0.5,
    )

    ax.set_title(name)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")  # type: ignore[attr-defined]
    ax.set_box_aspect([1.0, 1.0, 1.0])  # type: ignore[arg-type]
    plt.show()


def main():
    """Main function for running the simulator."""
    args = settings()
    if not os.path.exists(args.file):
        print(f"File not found: {args.file}")
        return

    if not os.path.isfile(args.file):
        print(f"Cannot read as file: {args.file}")
        return

    with open(file=args.file, encoding="utf-8") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(f"Exception: {exc}")
            print(f"Cannot parse: {args.file}")
            return

    mass, pos, vel = compose(config=config)  # pylint: disable=unused-variable
    plot(name=config[cc.SIMULATION][cc.NAME], pos=pos, vel=vel, subsample=5000)


if __name__ == "__main__":
    main()
