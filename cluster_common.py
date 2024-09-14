"""
cluster_common.py

Set of common elements useful for the cluster generation.
"""

from typing import Dict, Any
import pint

A = "a"
ANGLE_DISTRIBUTION = "angle_distribution"
AXIAL_DISTRIBUTION = "axial_distribution"
AXIS = "axis"
B = "b"
C = "c"
CLUSTERS = "clusters"
COMPONENT_MOTION = "component_motion"
COMPONENTS = "components"
DISTANCE = "distance"
ELLIPSOID_SHAPE = "ellipsoid_shape"
FREQUENCY = "frequency"
GRAVITATIONAL_CONST = 6.67430e-11
G = "g"
GROWTH = "growth"
MASS_DISTRIBUTION = "mass_distribution"
MEAN = "mean"
MOTION = "motion"
NAME = "name"
OFFSET = "offset"
OFFSET_DISTRIBUTION = "offset_distribution"
ORIENTATION = "orientation"
PHASE = "phase"
POSITION_DISTRIBUTION = "position_distribution"
R = "r"
RADIAL_DISTRIBUTION = "radial_distribution"
RADIUS = "radius"
RENDER = "render"
SCALE = "scale"
SIMULATION = "simulation"
STAR_COUNT = "star_count"
STDEV = "stdev"
STEPS = "steps"
TIME_STEP = "time_step"
THETA = "theta"
UNITS = "units"
VALUE = "value"
X = "x"
Y = "y"
Z = "z"


DEFAULT_MOTION = {
    X: {VALUE: 0.0, UNITS: "km/s"},
    Y: {VALUE: 0.0, UNITS: "km/s"},
    Z: {VALUE: 0.0, UNITS: "km/s"},
}

DEFAULT_RENDER = {"r": 1.0, "g": 1.0, "b": 1.0, "radius": 1.0}

DEFAULT_OFFSET = {
    X: {VALUE: 0.0, UNITS: "m"},
    Y: {VALUE: 0.0, UNITS: "m"},
    Z: {VALUE: 0.0, UNITS: "m"},
}

DEFAULT_ORIENTATION = {
    AXIS: {X: 0.0, Y: 0.0, Z: 1.0},
    THETA: {VALUE: 0.0, UNITS: "radians"},
}

DEFAULT_PHASE = {VALUE: 0.0, UNITS: "radian"}

ureg = pint.UnitRegistry()
ureg.define("solar_mass=1.9884E30 kilograms=sol")
quant = ureg.Quantity


def validate(config: Dict[str, Any], keys: set) -> None:
    """
    Validate that the dictionary keys don't contain unknown values

    Args:
        config - Configuration dict to validate
        keys - Valid keys to check against the config dict
    """
    keys_present = set(config.keys())
    unknown_keys = keys_present - keys
    if len(unknown_keys) > 0:
        raise KeyError(
            f"Unknown keys {unknown_keys} in {config}; expected keys: {keys}."
        )
