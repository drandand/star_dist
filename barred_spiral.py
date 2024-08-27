import numpy as np
import matplotlib.pyplot as plt

# from mpl_toolkits.mplot3d import Axes3D


def generate_bar(N_bar, length, width, thickness, v_max):
    """
    Generate the central bar component of the galaxy.

    Parameters:
    - N_bar: Number of stars in the bar
    - length: Length of the bar (kpc)
    - width: Width of the bar (kpc)
    - thickness: Thickness of the bar (kpc)
    - v_max: Maximum rotational velocity (km/s)

    Returns:
    - positions: (N_bar, 3) array of positions
    - velocities: (N_bar, 3) array of velocities
    """
    # Positions
    x = np.random.uniform(-length / 2, length / 2, N_bar)
    y = np.random.normal(0, width / 2, N_bar)
    z = np.random.normal(0, thickness / 2, N_bar)
    positions = np.column_stack((x, y, z))

    # Velocities
    # Assuming stars in the bar have some rotation and random motions
    radius = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    v_rot = v_max * (radius / (length / 2))  # Linear increase up to edge of bar
    v_rot = np.clip(v_rot, -v_max, v_max)

    vx = -v_rot * np.sin(theta) + np.random.normal(0, v_max * 0.1, N_bar)
    vy = v_rot * np.cos(theta) + np.random.normal(0, v_max * 0.1, N_bar)
    vz = np.random.normal(0, v_max * 0.05, N_bar)

    velocities = np.column_stack((vx, vy, vz))

    return positions, velocities


def generate_spiral_arms(
    N_arms, N_per_arm, R_min, R_max, v_max, arm_width, arm_spread, arm_pitch_angle
):
    """
    Generate the spiral arms of the galaxy.

    Parameters:
    - N_arms: Number of spiral arms
    - N_per_arm: Number of stars per arm
    - R_min: Minimum radius of the arms (kpc)
    - R_max: Maximum radius of the arms (kpc)
    - v_max: Maximum rotational velocity (km/s)
    - arm_width: Width of the arms (kpc)
    - arm_spread: Degree of scatter around the arm (kpc)
    - arm_pitch_angle: Pitch angle of the arms (degrees)

    Returns:
    - positions: (N_arms*N_per_arm, 3) array of positions
    - velocities: (N_arms*N_per_arm, 3) array of velocities
    """
    positions = []
    velocities = []

    pitch_rad = np.deg2rad(arm_pitch_angle)
    # theta_max = (R_max - R_min) / np.tan(pitch_rad)

    for i in range(N_arms):
        # Generate radii and theta for each arm
        radii = np.linspace(R_min, R_max, N_per_arm)
        theta = radii / np.tan(pitch_rad) + i * (2 * np.pi / N_arms)
        theta += np.random.normal(0, arm_spread, N_per_arm)

        # Convert to Cartesian coordinates
        x = radii * np.cos(theta) + np.random.normal(0, arm_width, N_per_arm)
        y = radii * np.sin(theta) + np.random.normal(0, arm_width, N_per_arm)
        z = np.random.normal(0, arm_width / 2, N_per_arm)

        # Calculate velocities assuming circular orbits
        v = v_max * (radii / R_max) ** 0.5  # Simple rotation curve
        vx = -v * np.sin(theta)
        vy = v * np.cos(theta)
        vz = np.random.normal(0, v_max * 0.05, N_per_arm)

        positions.append(np.column_stack((x, y, z)))
        velocities.append(np.column_stack((vx, vy, vz)))

    positions = np.vstack(positions)
    velocities = np.vstack(velocities)

    return positions, velocities


def generate_disk(N_disk, R_max, thickness, v_max):
    """
    Generate the disk component of the galaxy.

    Parameters:
    - N_disk: Number of stars in the disk
    - R_max: Maximum radius of the disk (kpc)
    - thickness: Thickness of the disk (kpc)
    - v_max: Maximum rotational velocity (km/s)

    Returns:
    - positions: (N_disk, 3) array of positions
    - velocities: (N_disk, 3) array of velocities
    """
    # Generate radii with exponential distribution
    radii = np.random.exponential(scale=R_max / 3, size=N_disk)
    radii = radii[radii <= R_max]

    # Random theta
    theta = np.random.uniform(0, 2 * np.pi, len(radii))

    # Positions
    x = radii * np.cos(theta)
    y = radii * np.sin(theta)
    z = np.random.normal(0, thickness / 2, len(radii))

    positions = np.column_stack((x, y, z))

    # Velocities
    v = v_max * (radii / R_max) ** 0.5
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


def generate_barred_spiral_galaxy(params):
    """
    Generate a barred spiral galaxy by combining bar, spiral arms, disk, and bulge.

    Parameters:
    - params: Dictionary containing all necessary parameters.

    Returns:
    - positions: (N_total, 3) array of positions
    - velocities: (N_total, 3) array of velocities
    """
    positions = []
    velocities = []

    # Generate bar
    pos_bar, vel_bar = generate_bar(
        N_bar=params["N_bar"],
        length=params["bar_length"],
        width=params["bar_width"],
        thickness=params["bar_thickness"],
        v_max=params["v_max"],
    )
    positions.append(pos_bar)
    velocities.append(vel_bar)

    # Generate spiral arms
    pos_arms, vel_arms = generate_spiral_arms(
        N_arms=params["N_arms"],
        N_per_arm=params["N_per_arm"],
        R_min=params["arm_R_min"],
        R_max=params["arm_R_max"],
        v_max=params["v_max"],
        arm_width=params["arm_width"],
        arm_spread=params["arm_spread"],
        arm_pitch_angle=params["arm_pitch_angle"],
    )
    positions.append(pos_arms)
    velocities.append(vel_arms)

    # Generate disk
    pos_disk, vel_disk = generate_disk(
        N_disk=params["N_disk"],
        R_max=params["disk_R_max"],
        thickness=params["disk_thickness"],
        v_max=params["v_max"],
    )
    positions.append(pos_disk)
    velocities.append(vel_disk)

    # Generate bulge
    if params.get("N_bulge", 0) > 0:
        pos_bulge, vel_bulge = generate_bulge(
            N_bulge=params["N_bulge"],
            radius=params["bulge_radius"],
            v_disp=params["bulge_v_disp"],
        )
        positions.append(pos_bulge)
        velocities.append(vel_bulge)

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

    ax.set_title("Barred Spiral Galaxy")
    ax.set_xlabel("X (kpc)")
    ax.set_ylabel("Y (kpc)")
    ax.set_zlabel("Z (kpc)")
    ax.set_box_aspect([1, 1, 0.5])
    plt.show()


def main():
    # Define parameters
    params = {
        "N_bar": 5000,
        "bar_length": 5,
        "bar_width": 1,
        "bar_thickness": 0.5,
        "N_arms": 2,
        "N_per_arm": 5000,
        "arm_R_min": 5,
        "arm_R_max": 15,
        "arm_width": 0.5,
        "arm_spread": 0.2,
        "arm_pitch_angle": 15,
        "N_disk": 10000,
        "disk_R_max": 20,
        "disk_thickness": 0.5,
        "N_bulge": 2000,
        "bulge_radius": 2,
        "bulge_v_disp": 100,
        "v_max": 200,
    }

    # Generate galaxy
    positions, velocities = generate_barred_spiral_galaxy(params)

    # Plot galaxy
    plot_galaxy(positions, velocities, subsample=2000)


if __name__ == "__main__":
    main()
