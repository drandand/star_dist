import numpy as np
import matplotlib.pyplot as plt

# from mpl_toolkits.mplot3d import Axes3D


def generate_spiral_galaxy_3d(
    N, R_max, v_max, z_scale=0.1, num_arms=2, arm_tightness=0.05, spread=0.1
):
    """
    Generate a random spiral galaxy with N stars in 3D.

    Parameters:
    - N: Number of stars
    - R_max: Maximum radius of the galaxy
    - v_max: Maximum velocity for stars
    - z_scale: Scale height for the vertical distribution of stars
    - num_arms: Number of spiral arms
    - arm_tightness: Tightness of the spiral arms
    - spread: Spread of stars around the spiral arms

    Returns:
    - positions: Nx3 array of star positions (x, y, z)
    - velocities: Nx3 array of star velocities (vx, vy, vz)
    """

    # Random radii following an exponential disk distribution
    radii = np.random.exponential(scale=R_max / 4, size=N)
    radii = radii[radii < R_max]

    # Generate angles theta for spiral arm structure
    theta = np.random.uniform(0, 2 * np.pi, size=radii.shape)

    # Spiral arm perturbation
    arm_offset = np.sin(num_arms * theta) * np.exp(-radii * arm_tightness)

    # Apply the perturbation to theta
    theta += arm_offset * spread

    # Vertical distribution (Gaussian or exponential)
    z = np.random.normal(scale=z_scale * R_max, size=radii.shape)

    # Convert polar coordinates to Cartesian coordinates
    x = radii * np.cos(theta)
    y = radii * np.sin(theta)
    positions = np.column_stack((x, y, z))

    # Velocities: Assume a flat rotation curve (v_max) for simplicity
    v = v_max * np.ones_like(radii)

    # Calculate velocity components
    vx = -v * np.sin(theta)
    vy = v * np.cos(theta)

    # Vertical velocity component (small random motion in z)
    vz = np.random.normal(scale=v_max * 0.1, size=radii.shape)

    velocities = np.column_stack((vx, vy, vz))

    return positions, velocities


def main():
    # Parameters
    N = 10000  # Number of stars
    R_max = 15  # Maximum radius (kpc)
    v_max = 220  # Maximum velocity (km/s)
    z_scale = 0.1  # Scale height for vertical distribution
    num_arms = 2  # Number of spiral arms
    arm_tightness = 1.0  # Tightness of the spiral arms
    spread = 0.1  # Spread of stars around the spiral arms

    # Generate the galaxy
    positions, velocities = generate_spiral_galaxy_3d(
        N, R_max, v_max, z_scale, num_arms, arm_tightness, spread
    )

    # Plot the 3D positions
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=1, color="blue")
    ax.set_title("Spiral Galaxy - Star Positions in 3D")
    ax.set_xlabel("x (kpc)")
    ax.set_ylabel("y (kpc)")
    ax.set_zlabel("z (kpc)")
    ax.set_box_aspect([1, 1, 0.5])  # Aspect ratio
    plt.show()

    # Plot the 3D velocities
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.quiver(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        velocities[:, 0],
        velocities[:, 1],
        velocities[:, 2],
        length=0.5,
        color="red",
    )
    ax.set_title("Spiral Galaxy - Star Velocities in 3D")
    ax.set_xlabel("x (kpc)")
    ax.set_ylabel("y (kpc)")
    ax.set_zlabel("z (kpc)")
    ax.set_box_aspect([1, 1, 0.5])  # Aspect ratio
    plt.show()


if __name__ == "__main__":
    main()
