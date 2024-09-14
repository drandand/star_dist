"""
cluster_component.py

Contains the class definition for the Component class.
A single cluster is composed of one or more components which
describes the position, mass and velocity of a subset of the
cluster stars.
"""

from typing import Dict, Any
import numpy as np

import cluster_parser as cp
import cluster_distributions as cd
import cluster_common as cc


class Component:
    """
    Component class defining how individual cluster elements are structured

    Args:
        config - Dictionary containing component element specifications
    """

    def __init__(self, config: Dict[str, Any]):
        cc.validate(
            config=config,
            keys={
                cc.NAME,
                cc.STAR_COUNT,
                cc.POSITION_DISTRIBUTION,
                cc.MASS_DISTRIBUTION,
                cc.COMPONENT_MOTION,
                cc.RENDER,
            },
        )
        self._name: str = cp.select(src=config, key=cc.NAME)
        self._star_count: int = cp.select(src=config, key=cc.STAR_COUNT)
        self._position_distribution = cd.vector_distribution(
            config=cp.select(src=config, key=cc.POSITION_DISTRIBUTION)
        )
        self._mass_distribution = cd.scalar_distribution(
            config=cp.select(src=config, key=cc.MASS_DISTRIBUTION)
        )
        self._component_motion = cd.vector_distribution(
            config=cp.select(src=config, key=cc.COMPONENT_MOTION)
        )
        self._render = cp.Render(
            cp.select(
                src=config,
                key=cc.RENDER,
                default=cc.DEFAULT_RENDER,
                required=False,
            )
        )

    @property
    def name(self) -> str:
        """Property containing the name of the component"""
        return self._name

    @property
    def star_count(self) -> int:
        """Property containing the number of stars in the component"""
        return self._star_count

    @property
    def position_distribution(self) -> cd.VectorDistribution:
        """
        Property for describing how the stars are distributed within the
        component
        """
        return self._position_distribution

    @property
    def mass_distribution(self) -> cd.ScalarDistribution:
        """Property for the probability distribution for the star mass"""
        return self._mass_distribution

    @property
    def component_motion(self) -> cd.VectorDistribution:
        """
        Property describing how the component as a whole will move above and
        beyond how they are initially affected by gravitational forces for
        each star
        """
        return self._component_motion

    @property
    def render(self) -> cp.Render:
        """Property describing how to render the stars in this component"""
        return self._render

    def gen_masses(self, units: str) -> np.ndarray:
        """
        Generate the star masses using the mass distribution

        Args:
            units - Units (e.g. kg, solar masses) to use for the mass
            distribution

        Return:
            Numpy array containing mass of stars in the given units using the
            distribution
        """
        return self._mass_distribution.gen(count=self._star_count, units=units)

    def gen_positions(self, units: str) -> np.ndarray:
        """
        Generate the star positions using the position distribution

        Args:
            units - Units (e.g. kg, solar masses) to use for the mass
            distribution

        Return:
            Numpy array containing position of stars in the given units using
            the position distribution
        """
        return self._position_distribution.gen(
            count=self._star_count,
            units=units,
        )

    def gen_velocities(
        self,
        positions: np.ndarray,
        masses: np.ndarray,
        com: np.ndarray,
        mass: float,
        units: str,
    ) -> np.ndarray:
        """
        Generate the velocities for stars the using the velocity distribution

        Args:
            positions - Positions of the stars to use for computing the
            velocities
            masses - Masses for the stars to use for computing the velocities
            com - Center of mass of the cluster
            mass - Total mass of the cluster
            units - Units for the velocities

        Return:
            Return a numpy array of the velocities of the component
        """
        d = positions - com
        dist = np.linalg.norm(d, axis=1, keepdims=True)
        direction = np.cross(d / dist, np.array([0.0, 0.0, 1.0]))
        direction = direction / np.linalg.norm(
            direction,
            axis=1,
            keepdims=True,
        )
        v_rel = np.sqrt(cc.GRAVITATIONAL_CONST * mass / dist).flatten()
        vel = ((mass - masses) * v_rel / mass).reshape(-1, 1)
        return (vel * direction) + self._component_motion.gen(
            count=self._star_count, units=units
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the component to its equivalent dictionary

        Return:
            Dictionary representation of the component configuration
        """
        return {
            cc.NAME: self._name,
            cc.STAR_COUNT: self._star_count,
            cc.POSITION_DISTRIBUTION: self._position_distribution.to_dict(),
            cc.MASS_DISTRIBUTION: self._mass_distribution.to_dict(),
            cc.COMPONENT_MOTION: self._component_motion.to_dict(),
            cc.RENDER: self._render.to_dict(),
        }
