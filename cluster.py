"""
cluster.py

Contains classes and functions used to define high level features of a single
cluster.
"""

from typing import Dict, Any, Tuple, List
import numpy as np

import cluster_common as cc
import cluster_parser as cp
import cluster_component as co


def get_center_of_mass(
    masses: List[np.ndarray], positions: List[np.ndarray]
) -> Tuple[float, np.ndarray]:
    """
    Compute the center of mass of a collection of point masses.

    Args:
        masses - List of masses for each object
        positions - List of 3D vectors representing the location of each point
        mass

    Return:
        Tuple containing the total mass and the position of the mass center
        of all the point masses given.
    """
    all_m = np.concatenate(masses).reshape(-1, 1)
    all_p = np.concatenate(positions)
    total_mass = all_m.sum()
    center_of_mass = (all_m * all_p).sum(axis=0) / total_mass

    return total_mass, center_of_mass


class Cluster:
    """
    Class containing the high level features of a star cluster.

    Args:
        Dictionary containing the features of the star cluster
    """

    def __init__(self, config: dict):
        cc.validate(
            config=config,
            keys={
                cc.NAME,
                cc.OFFSET,
                cc.MOTION,
                cc.ORIENTATION,
                cc.COMPONENTS,
            },
        )
        self._name = cp.select(src=config, key=cc.NAME)
        self._offset = cp.UnitsVector3d(
            cp.select(
                src=config,
                key=cc.OFFSET,
                default=cc.DEFAULT_OFFSET,
                required=False,
            )
        )
        self._motion = cp.UnitsVector3d(
            cp.select(
                src=config,
                key=cc.MOTION,
                default=cc.DEFAULT_MOTION,
                required=False,
            )
        )
        self._orientation = cp.Orientation(
            cp.select(
                src=config,
                key=cc.ORIENTATION,
                default=cc.DEFAULT_ORIENTATION,
                required=False,
            )
        )
        self._components = [
            co.Component(x) for x in cp.select(src=config, key=cc.COMPONENTS)
        ]

    @property
    def name(self) -> str:
        """Accessor for the cluster name"""
        return self._name

    @property
    def offset(self) -> cp.UnitsVector3d:
        """Accessor for the cluster position offset"""
        return self._offset

    @property
    def motion(self) -> cp.UnitsVector3d:
        """Accessor for the cluster velocity"""
        return self._motion

    @property
    def orientation(self) -> cp.Orientation:
        """Accessor for the orientation of the cluster"""
        return self._orientation

    @property
    def components(self) -> List[co.Component]:
        """Accessor for the component specifications for the cluster"""
        return self._components

    def gen_masses(self, units: str) -> List[np.ndarray]:
        """
        Generate the star masses for each of component of the cluster

        Args:
            units - Units to use for computing each star mass

        Returns:
            List of numpy arrays containing the star masses for each component
            comprising the cluster
        """
        return [x.gen_masses(units=units) for x in self._components]

    def gen_positions(self, units: str) -> List[np.ndarray]:
        """
        Generate the star position for each component in the cluster

        Args:
            units - Units to use for computing each star position

        Returns:
            List of numpy arrays containing the star positions for each
            component comprising the cluster
        """
        return [x.gen_positions(units=units) for x in self._components]

    def gen_velocities(
        self,
        masses: List[np.ndarray],
        positions: List[np.ndarray],
        units: str,
    ) -> List[np.ndarray]:
        """
        Generate the star velocity for each component in the cluster

        Args:
            mass - Lists of star masses for each component
            positions - List of star positions for each component
            units - Units to represent the velocity of the stars

        Returns:
            List of numpy arrays containing the star velocity for each
            component comprising the cluster
        """
        total_mass, center_of_mass = get_center_of_mass(
            masses=masses,
            positions=positions,
        )

        return [
            x.gen_velocities(
                positions=p,
                masses=m,
                com=center_of_mass,
                mass=total_mass,
                units=units,
            )
            for x, m, p in zip(self._components, masses, positions)
        ]

    def gen_cluster(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate the stars in the cluster and return arrays containing the
        different star masses, position vectors and velocity vectors.

        Returns:
            Tuple containing the mass, position and velocity of all the stars
            in the cluster.
        """
        masses = self.gen_masses(units="kg")
        positions = self.gen_positions(units="meter")
        velocities = self.gen_velocities(
            masses=masses, positions=positions, units="meter / second"
        )
        rot = self._orientation.rotation()
        offset = self._offset.to_numpy(units="meters")
        motion = self._motion.to_numpy(units="meters / sec")

        ret_masses = np.concatenate(masses)
        ret_positions = np.dot(np.concatenate(positions), rot) + offset
        ret_velocities = np.dot(np.concatenate(velocities), rot) + motion
        return ret_masses, ret_positions, ret_velocities

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the cluster parameters into a dictionary object

        Returns:
            Dictionary containing the elements of the previous class instance
        """
        return {
            cc.NAME: self._name,
            cc.OFFSET: self._offset.to_dict(),
            cc.MOTION: self._motion.to_dict(),
            cc.ORIENTATION: self._orientation.to_dict(),
            cc.COMPONENTS: [x.to_dict() for x in self._components],
        }
