import numpy as np
import matplotlib.pyplot as plt

# from mpl_toolkits.mplot3d import Axes3D


def generate_lenticular_disk(N_disk, R_max, thickness, v_max):
    """
    Generate the disk component of a lenticular galaxy.

    Parameters:
    - N_disk: Number of stars in the disk
    - R_max: Maximum radius of the disk (kpc)
    - thickness: Thickness of the disk (kpc)
    - v_max: Maximum rotational velocity (km/s)

    Returns:
    - positions: (N_disk, 3) array of positions
    - velocities: (N_disk, 3) array of velocities
    """
    # Generate radii with an exponential distribution
    radii = np.random.exponential(scale=R_max / 3, size=N_disk)
    radii = radii[radii <= R_max]  # Ensure radii don't exceed R_max

    # Random theta
    theta = np.random.uniform(0, 2 * np.pi, len(radii))

    # Positions
    x = radii * np.cos(theta)
    y = radii * np.sin(theta)
    z = np.random.normal(0, thickness / 2, len(radii))

    positions = np.column_stack((x, y, z))

    # Velocities
    v = v_max * (radii / R_max) ** 0.5  # Simple rotation curve
    vx = -v * np.sin(theta) + np.random.normal(0, v_max * 0.05, len(radii))
    vy = v * np.cos(theta) + np.random.normal(0, v_max * 0.05, len(radii))
    vz = np.random.normal(0, v_max * 0.02, len(radii))

    velocities = np.column_stack((vx, vy, vz))

    return positions, velocities


def generate_bulge(N_bulge, radius, v_disp):
    """
    Generate the central bulge component of the galaxy.

    Parameters:
    - N_bulge: Number of stars in the bulge
    - radius: Radius of the bulge (kpc)
    - v_disp: Velocity dispersion in the bulge (km/s)

    Returns:
    - positions: (N_bulge, 3) array of positions
    - velocities: (N_bulge, 3) array of velocities
    """
    # Positions - spherical distribution
    r = np.random.uniform(0, radius, N_bulge) ** (1 / 3)
    theta = np.arccos(np.random.uniform(-1, 1, N_bulge))
    phi = np.random.uniform(0, 2 * np.pi, N_bulge)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    positions = np.column_stack((x, y, z))

    # Velocities - random isotropic distribution
    vx = np.random.normal(0, v_disp, N_bulge)
    vy = np.random.normal(0, v_disp, N_bulge)
    vz = np.random.normal(0, v_disp, N_bulge)

    velocities = np.column_stack((vx, vy, vz))

    return positions, velocities


def generate_lenticular_galaxy(params):
    """
    Generate a lenticular galaxy by combining the disk and bulge components.

    Parameters:
    - params: Dictionary containing all necessary parameters.

    Returns:
    - positions: (N_total, 3) array of positions
    - velocities: (N_total, 3) array of velocities
    """
    positions = []
    velocities = []

    # Generate bulge
    pos_bulge, vel_bulge = generate_bulge(
        N_bulge=params["N_bulge"],
        radius=params["bulge_radius"],
        v_disp=params["bulge_v_disp"],
    )
    positions.append(pos_bulge)
    velocities.append(vel_bulge)

    # Generate disk
    pos_disk, vel_disk = generate_lenticular_disk(
        N_disk=params["N_disk"],
        R_max=params["disk_R_max"],
        thickness=params["disk_thickness"],
        v_max=params["v_max"],
    )
    positions.append(pos_disk)
    velocities.append(vel_disk)

    # Combine all components
    positions = np.vstack(positions)
    velocities = np.vstack(velocities)

    return positions, velocities


def plot_galaxy(positions, velocities, subsample=1000):
    """
    Plot the galaxy positions and velocities in 3D.

    Parameters:
    - positions: (N, 3) array of positions
    - velocities: (N, 3) array of velocities
    - subsample: Number of stars to plot for velocities to avoid clutter
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot positions
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        s=0.5,
        alpha=0.6,
        color="cyan",
    )

    # Plot velocity vectors (subsampled)
    idx = np.random.choice(len(positions), size=subsample, replace=False)
    ax.quiver(
        positions[idx, 0],
        positions[idx, 1],
        positions[idx, 2],
        velocities[idx, 0],
        velocities[idx, 1],
        velocities[idx, 2],
        length=0.5,
        normalize=True,
        color="red",
        alpha=0.5,
    )

    ax.set_title("Lenticular Galaxy")
    ax.set_xlabel("X (kpc)")
    ax.set_ylabel("Y (kpc)")
    ax.set_zlabel("Z (kpc)")
    ax.set_box_aspect([1, 1, 0.5])
    plt.show()


def main():

    # Define parameters
    params = {
        "N_disk": 20000,
        "disk_R_max": 20,
        "disk_thickness": 1,
        "N_bulge": 5000,
        "bulge_radius": 3,
        "bulge_v_disp": 150,
        "v_max": 220,
    }

    # Generate galaxy
    positions, velocities = generate_lenticular_galaxy(params)

    # Plot galaxy
    plot_galaxy(positions, velocities, subsample=3000)


if __name__ == "__main__":
    main()
