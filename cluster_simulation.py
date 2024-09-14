"""
cluster_simulation.py

The simulator module contains the main configuration information defining the
location and mass distributions for the stars in the simulation.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from pint.facets.plain import PlainQuantity

import cluster as cl
import cluster_common as cc
import cluster_parser as cp


class Simulation:
    """
    Class to define the overall distribution of stars and define their mass
    distributions for the entire simulation, which could consist of multiple
    clusters.

    Args:
        config: Dictionary containing the cluster configuration details
    """

    def __init__(self, config: dict):
        cc.validate(
            config=config,
            keys={
                cc.NAME,
                cc.TIME_STEP,
                cc.STEPS,
                cc.CLUSTERS,
            },
        )
        self._name: str = cp.select(src=config, key=cc.NAME)
        self._time_step = cp.parse_unit(config[cc.TIME_STEP])
        self._steps: int = int(cp.select(src=config, key=cc.STEPS))
        clusters = cp.select(src=config, key=cc.CLUSTERS)
        self._clusters = [cl.Cluster(config=cfg) for cfg in clusters]

    @property
    def name(self) -> str:
        """
        Property containing the name of the simulation
        """
        return self._name

    @property
    def time_step(self) -> PlainQuantity:
        """
        Property containing the span of time for each step in the simulation.
        """
        return self._time_step

    @property
    def steps(self) -> int:
        """Property containing the number of steps to run the simulation."""
        return self._steps

    @property
    def clusters(self) -> List[cl.Cluster]:
        """Property containing the list of clusters for the simulation"""
        return self._clusters

    def gen_clusters(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate all of the clusters and return the mass, position and
        velocity of the stars in each cluster as a list of tuples.

        Return:
            List of tuples where the first element of each tuple is the mass,
            the second is the position and the third is the velocity
        """
        return [x.gen_cluster() for x in self._clusters]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the simulation and all of its elements into a dictionary which
        can be shared with other implementations.

        Return:
            Dictionary containing the configuration properties of the entire
            simulation
        """
        return {
            cc.TIME_STEP: cp.to_dict(quantity=self._time_step),
            cc.STEPS: self._steps,
            cc.CLUSTERS: [x.to_dict() for x in self._clusters],
        }
