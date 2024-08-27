import numpy as np
import matplotlib.pyplot as plt

# from mpl_toolkits.mplot3d import Axes3D


def generate_elliptical_galaxy(N, a, b, c, v_disp):
    """
    Generate a random elliptical galaxy with N stars in 3D.

    Parameters:
    - N: Number of stars
    - a: Semi-major axis (kpc)
    - b: Semi-minor axis 1 (kpc)
    - c: Semi-minor axis 2 (kpc)
    - v_disp: Velocity dispersion (km/s)

    Returns:
    - positions: Nx3 array of star positions (x, y, z)
    - velocities: Nx3 array of star velocities (vx, vy, vz)
    """

    # Generate positions using a triaxial Gaussian distribution
    x = np.random.normal(scale=a, size=N)
    y = np.random.normal(scale=b, size=N)
    z = np.random.normal(scale=c, size=N)
    positions = np.column_stack((x, y, z))

    # Generate velocities using a Gaussian distribution (isotropic velocity dispersion)
    vx = np.random.normal(scale=v_disp, size=N)
    vy = np.random.normal(scale=v_disp, size=N)
    vz = np.random.normal(scale=v_disp, size=N)
    velocities = np.column_stack((vx, vy, vz))

    return positions, velocities


def main():
    # Parameters
    N = 10000  # Number of stars
    a = 10  # Semi-major axis (kpc)
    b = 7  # Semi-minor axis 1 (kpc)
    c = 5  # Semi-minor axis 2 (kpc)
    v_disp = 150  # Velocity dispersion (km/s)

    # Generate the elliptical galaxy
    positions, velocities = generate_elliptical_galaxy(N, a, b, c, v_disp)

    # Plot the 3D positions
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=1, color="green")
    ax.set_title("Elliptical Galaxy - Star Positions in 3D")
    ax.set_xlabel("x (kpc)")
    ax.set_ylabel("y (kpc)")
    ax.set_zlabel("z (kpc)")
    ax.set_box_aspect([1, 1, c / a])  # Aspect ratio
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
        length=0.1,
        color="red",
    )
    ax.set_title("Elliptical Galaxy - Star Velocities in 3D")
    ax.set_xlabel("x (kpc)")
    ax.set_ylabel("y (kpc)")
    ax.set_zlabel("z (kpc)")
    ax.set_box_aspect([1, 1, c / a])  # Aspect ratio
    plt.show()


if __name__ == "__main__":
    main()
