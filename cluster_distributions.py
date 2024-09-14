"""
cluster_distributions.py

Module contains a collection of scalar and vector distributions used to
generate random scalar and vector values.
"""

from enum import Enum
from typing import Dict, Any
from abc import ABC, abstractmethod

from pint.facets.plain import PlainQuantity
import numpy as np

import cluster_parser as cp
import cluster_common as cc


class ScalarDistributionType(Enum):
    """
    Enumeration describing different types of generators for different
    probability distributions.
    FIXED_SCALAR: Constant value with no variability
    EXPONENTIAL: Exponential distribution
    NORMAL: Normal (i.e. Gaussian) distribution
    LOGNORMAL: Log-normal distribution
    """

    FIXED_SCALAR = 0
    EXPONENTIAL = 1
    NORMAL = 2
    LOGNORMAL = 3


class ScalarDistribution(ABC):
    """
    Abstract class defining the structure of scalar distribution
    types.

    Args:
        dist_type: Type of distribution instantiated
    """

    def __init__(self, dist_type: ScalarDistributionType):
        self._dist_type: ScalarDistributionType = dist_type

    @property
    def dist_type(self) -> ScalarDistributionType:
        """Property of the scalar distribution type"""
        return self._dist_type

    @abstractmethod
    def gen(self, count: int, units: str) -> np.ndarray:
        """
        Generate some number of random values storing them in a numpy array.

        Args:
            count: Number of random values to generate
            units: Units to use when generating those values

        Return:
            numpy array with 'count' values derived from the underlying
            probability distribution and adjusted with the types of units
            given.
        """

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the scalar distribution into a dictionary value

        Return:
            Dictionary containing the parameters of the distribution
        """


class FixedScalar(ScalarDistribution):
    """
    Class defining a fixed scalar value.

    Args:
        config: Configuration defining the FixedScalar distribution
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(ScalarDistributionType.FIXED_SCALAR)
        self._value = cp.parse_unit(val=config)

    @property
    def value(self) -> PlainQuantity:
        """Property containing the value of the fixed scalar distribution"""
        return self._value

    def gen(self, count: int, units: str) -> np.ndarray:
        """
        Generate some number of fixed scalar values storing them in a numpy
        array.

        Args:
            count: Number of fixed scalar values to generate
            units: Units to use when generating those values

        Return:
            numpy array with 'count' values derived from the underlying
            fixed scalar and adjusted with the types of units
            given.
        """
        return np.full(count, self._value.to(units).magnitude)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the distribution into a dictionary value

        Return:
            Dictionary containing the parameters of the distribution
        """
        return {
            self._dist_type.name.lower(): cp.to_dict(quantity=self._value),
        }


class Exponential(ScalarDistribution):
    """
    Class definition for am exponential distribution

    Args:
        config: Dictionary containing the definition for
        this exponential distribution class instance
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(ScalarDistributionType.EXPONENTIAL)
        cc.validate(config=config, keys={cc.SCALE})
        self._scale = cp.parse_unit(cp.select(src=config, key=cc.SCALE))

    @property
    def scale(self) -> PlainQuantity:
        """
        Property containing the scale parameter for an exponential
        distribution.
        """
        return self._scale

    def gen(self, count: int, units: str) -> np.ndarray:
        """
        Generate some number of exponentially distributed values storing them
        in a numpy array.

        Args:
            count: Number of exponentially distributed values to generate
            units: Units to use when generating those values

        Return:
            numpy array with 'count' values derived from the underlying
            exponential distribution and adjusted with the types of units
            given.
        """
        scale = self._scale.to(cc.ureg(units)).magnitude
        return np.random.exponential(scale=scale, size=count)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the distribution into a dictionary value

        Return:
            Dictionary containing the parameters of the distribution
        """
        return {
            self._dist_type.name.lower(): {
                cc.SCALE: cp.to_dict(quantity=self._scale),
            }
        }


class Normal(ScalarDistribution):
    """
    Class containing the basis for a generator for normally distributed random
    values.

    Args:
        config: Dictionary containing the configuration information for the
        normal distribution
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(ScalarDistributionType.NORMAL)
        cc.validate(config=config, keys={cc.MEAN, cc.STDEV})
        self._mean = cp.parse_unit(cp.select(src=config, key=cc.MEAN))
        self._stdev = cp.parse_unit(cp.select(src=config, key=cc.STDEV))

    @property
    def mean(self) -> PlainQuantity:
        """Property containing the mean of the normal distribution."""
        return self._mean

    @property
    def stdev(self) -> PlainQuantity:
        """
        Property containing the standard deviation of the normal
        distribution.
        """
        return self._stdev

    def gen(self, count: int, units: str) -> np.ndarray:
        """
        Generate some number of normally distributed values storing them
        in a numpy array.

        Args:
            count: Number of normally distributed values to generate
            units: Units to use when generating those values

        Return:
            numpy array with 'count' values derived from the underlying normal
            distribution and adjusted with the types of units given.
        """
        loc = self._mean.to(cc.ureg(units)).magnitude
        scale = self._stdev.to(cc.ureg(units)).magnitude

        return np.random.normal(loc=loc, scale=scale, size=count)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the distribution into a dictionary value

        Return:
            Dictionary containing the parameters of the distribution
        """
        return {
            self._dist_type.name.lower(): {
                cc.MEAN: cp.to_dict(quantity=self._mean),
                cc.STDEV: cp.to_dict(quantity=self._mean),
            }
        }


class Lognormal(ScalarDistribution):
    """
    Class containing the basis for a generator for log-normally distributed
    random values.

    Args:
        config: Dictionary containing the configuration information for the
        log-normal distribution
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(ScalarDistributionType.LOGNORMAL)
        cc.validate(config=config, keys={cc.MEAN, cc.STDEV})
        self._mean = cp.parse_unit(cp.select(src=config, key=cc.MEAN))
        self._stdev = cp.parse_unit(cp.select(src=config, key=cc.STDEV))

    @property
    def mean(self) -> PlainQuantity:
        """Property containing the mean of the log-normal distribution."""
        return self._mean

    @property
    def stdev(self) -> PlainQuantity:
        """
        Property containing the standard deviation of the log-normal
        distribution.
        """
        return self._stdev

    def gen(self, count: int, units: str) -> np.ndarray:
        """
        Generate some number of log-normally distributed values storing them
        in a numpy array.

        Args:
            count: Number of log-normally distributed values to generate
            units: Units to use when generating those values

        Return:
            numpy array with 'count' values derived from the underlying
            log-normal distribution and adjusted with the types of units
            given.
        """
        mean = self._mean.to(cc.ureg(units)).magnitude
        stdev = self._stdev.to(cc.ureg(units)).magnitude

        return np.random.lognormal(mean=mean, sigma=stdev, size=count)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the distribution into a dictionary value

        Return:
            Dictionary containing the parameters of the distribution
        """
        return {
            self._dist_type.name.lower(): {
                cc.MEAN: cp.to_dict(quantity=self._mean),
                cc.STDEV: cp.to_dict(quantity=self._mean),
            }
        }


def scalar_distribution(config: Dict[str, Any]) -> ScalarDistribution:
    """
    Factory for creating scalar distribution classes derived from the given
    configuration.

    Args:
        config: Configuration used to derive the scalar distribution class

    Return:
        A scalar distribution derived from the given dictionary
    """
    if len(config) != 1:
        raise ValueError(f"scalar_distribution expected 1 key in {config}")
    key_name = list(config.keys())[0]
    if key_name.upper() not in set(x.name for x in ScalarDistributionType):
        raise ValueError(f"Unknown ScalarDistributionType {key_name.upper()}.")

    distribution_type = ScalarDistributionType[key_name.upper()]

    if distribution_type == ScalarDistributionType.FIXED_SCALAR:
        return FixedScalar(config=config[key_name])
    if distribution_type == ScalarDistributionType.EXPONENTIAL:
        return Exponential(config=config[key_name])
    if distribution_type == ScalarDistributionType.NORMAL:
        return Normal(config=config[key_name])
    if distribution_type == ScalarDistributionType.LOGNORMAL:
        return Lognormal(config=config[key_name])

    raise ValueError(f"Unknown ScalarDistributionType {key_name.upper()}.")


class VectorDistributionTypes(Enum):
    """
    Enumeration for different types methods for generating
    random 3-d vectors for the cluster components.
    FIXED_VECTOR: Represents a constant vector value
    ELLIPSOID: Represents a collection of vectors distributed as an ellipsoid
    CYLINDER: Represents a collection of vectors distributed as a cylinder
    SPIRAL: Represent a collection of vectors distributed as a spiral
    """

    FIXED_VECTOR = 0
    ELLIPSOID = 1
    CYLINDER = 2
    SPIRAL = 3


class VectorDistribution(ABC):
    """
    Abstract class defining the basic structure of the vector distribution
    classes.
    """

    def __init__(self, dist_type: VectorDistributionTypes):
        self._dist_type: VectorDistributionTypes = dist_type

    @property
    def dist_type(self) -> VectorDistributionTypes:
        """
        Property for retrieving the vector distribution type for this vector.
        """
        return self._dist_type

    @abstractmethod
    def gen(self, count: int, units: str) -> np.ndarray:
        """
        Generate some number of random 3D vectors values storing them in a
        numpy array.

        Args:
            count: Number of random 3D vectors to generate
            units: Units to use when generating those vectors

        Return:
            numpy array with 'count' 3D vectors derived from the underlying
            distribution and adjusted with the types of units given.
        """

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the vector generator arguments into a dictionary

        Return:
            Dictionary containing the parameters of the vector generator
        """


class FixedVector(VectorDistribution):
    """
    Class containing a constant valued vector with no variability

    Args:
        config: Dictionary containing the specification of the fixed value
        to use when generating the resultant vector values.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(VectorDistributionTypes.FIXED_VECTOR)
        self._vector = cp.UnitsVector3d(config=config)

    @property
    def vector(self) -> cp.UnitsVector3d:
        """
        Property containing the fixed vector this distribution represents.
        """
        return self._vector

    def gen(self, count: int, units: str) -> np.ndarray:
        """
        Generate some number of fixed value 3D vectors values storing them in
        a numpy array.

        Args:
            count: Number of constant 3D vectors to generate
            units: Units to use when generating those vectors

        Return:
            numpy array with 'count' 3D vectors derived from the underlying
            fixed value and adjusted with the types of units given.
        """
        vec = self._vector.to(units)
        row = np.array([vec.x.magnitude, vec.y.magnitude, vec.z.magnitude])

        return np.tile(row, (count, 1))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the vector generator arguments into a dictionary

        Return:
            Dictionary containing the parameters of the vector generator
        """
        return {self._dist_type.name.lower(): self._vector.to_dict()}


class Ellipsoid(VectorDistribution):
    """
    Class used to generate 3D vector values distributed in a the general shape
    of an ellipsoid.  If all the produced vectors are normalized in accordance
    with the ellipsoid parameters, they will be uniformly distributed on the
    surface of the ellipsoid.

    Args:
        config: Dictionary used to define the underlying ellipsoid and how
        to deviate from that surface
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(VectorDistributionTypes.ELLIPSOID)
        cc.validate(
            config=config,
            keys={
                cc.ELLIPSOID_SHAPE,
                cc.DISTANCE,
                cc.ORIENTATION,
            },
        )

        self._ellipsoid_shape = cp.Ellipsoid(
            config=cp.select(src=config, key=cc.ELLIPSOID_SHAPE)
        )
        self._distance = scalar_distribution(
            config=cp.select(src=config, key=cc.DISTANCE)
        )
        self._orientation = cp.Orientation(
            config=cp.select(
                src=config,
                key=cc.ORIENTATION,
                default=cc.DEFAULT_ORIENTATION,
                required=False,
            )
        )

    @property
    def ellipsoid_shape(self) -> cp.Ellipsoid:
        """Property containing the parameters of the underlying ellipsoid"""
        return self._ellipsoid_shape

    @property
    def distance(self) -> ScalarDistribution:
        """
        Property containing the distance distribution to induce variability
        into the distance each vector lies away from the center of the
        ellipsoid.
        """
        return self._distance

    @property
    def orientation(self) -> cp.Orientation:
        """
        Property containing the orientation of the cluster
        """
        return self._orientation

    def gen(self, count: int, units: str) -> np.ndarray:
        """
        Generate some number of random 3D vectors values shaped as as
        ellipsoid storing them in a numpy array.

        Args:
            count: Number of random 3D vectors to generate
            units: Units to use when generating those vectors

        Return:
            numpy array with 'count' 3D vectors derived from the ellipsoid
            shape and adjusted with the types of units given.
        """
        d = self._distance.gen(count, units)
        theta = np.random.uniform(low=0, high=2 * np.pi, size=count)
        cos_phi = np.random.uniform(low=-1, high=1, size=count)
        sin_phi = np.sqrt(1 - cos_phi**2)

        x = d * self._ellipsoid_shape.a * sin_phi * np.cos(theta)
        y = d * self._ellipsoid_shape.b * sin_phi * np.sin(theta)
        z = d * self._ellipsoid_shape.c * cos_phi

        points = np.vstack((x, y, z)).T
        return np.dot(points, self._orientation.rotation())

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the vector generator arguments into a dictionary

        Return:
            Dictionary containing the parameters of the vector generator
        """
        return {
            self._dist_type.name.lower(): {
                cc.ELLIPSOID_SHAPE: self._ellipsoid_shape.to_dict(),
                cc.DISTANCE: self._distance.to_dict(),
                cc.ORIENTATION: self._orientation.to_dict(),
            }
        }


class Cylinder(VectorDistribution):
    """
    Class describing a cylindrical base structure for distributing random
    vectors.  The axial and radial distances are randomly distributed, and when
    normalized to a constant value, will result in the vectors being uniformly
    distributed on the surface of a cylinder

    Args:
        config: Dictionary containing the parameters for generating the
        randomly distributed vectors.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(VectorDistributionTypes.CYLINDER)
        cc.validate(
            config=config,
            keys={
                cc.AXIAL_DISTRIBUTION,
                cc.RADIAL_DISTRIBUTION,
                cc.ORIENTATION,
            },
        )
        self._axial_distribution = scalar_distribution(
            config=cp.select(src=config, key=cc.AXIAL_DISTRIBUTION)
        )
        self._radial_distribution = scalar_distribution(
            config=cp.select(src=config, key=cc.RADIAL_DISTRIBUTION)
        )
        self._orientation = cp.Orientation(
            config=cp.select(
                src=config,
                key=cc.ORIENTATION,
                default=cc.DEFAULT_ORIENTATION,
                required=False,
            )
        )

    @property
    def axial_distribution(self) -> ScalarDistribution:
        """
        Property containing the distribution of vector elements along the axis
        of the cylinder
        """
        return self._axial_distribution

    @property
    def radial_distribution(self) -> ScalarDistribution:
        """
        Property containing the distribution of the vector elements for the
        radial distance of the cylinder.
        """
        return self._radial_distribution

    @property
    def orientation(self) -> cp.Orientation:
        """
        Property containing the orientation of the cylinder to apply to
        the vectors following their generation in the canonical form.
        """
        return self._orientation

    def gen(self, count: int, units: str) -> np.ndarray:
        """
        Generate some number of random 3D vectors values shaped as as
        cylinder storing them in a numpy array.

        Args:
            count: Number of random 3D vectors to generate
            units: Units to use when generating those vectors

        Return:
            numpy array with 'count' 3D vectors derived from the cylinder's
            shape and adjusted with the types of units given.
        """
        theta = np.random.uniform(low=0, high=2.0 * np.pi, size=count)
        radials = self._radial_distribution.gen(count=count, units=units)
        x = radials * np.cos(theta)
        y = radials * np.sin(theta)
        z = self._axial_distribution.gen(count=count, units=units)

        points = np.vstack((x, y, z)).T
        return np.dot(points, self._orientation.rotation())

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the vector generator arguments into a dictionary

        Return:
            Dictionary containing the parameters of the vector generator
        """
        return {
            self._dist_type.name.lower(): {
                cc.AXIAL_DISTRIBUTION: self._axial_distribution.to_dict(),
                cc.RADIAL_DISTRIBUTION: self._radial_distribution.to_dict(),
                cc.ORIENTATION: self._orientation.to_dict(),
            }
        }


class Spiral(VectorDistribution):
    """
    Class defining the distribution of random 3D vectors in a roughly spiral
    shape.

    Args:
        config: Dictionary containing the parameters defining the structure
        of the spiral distribution.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(VectorDistributionTypes.SPIRAL)
        cc.validate(
            config=config,
            keys={
                cc.PHASE,
                cc.FREQUENCY,
                cc.GROWTH,
                cc.ANGLE_DISTRIBUTION,
                cc.OFFSET_DISTRIBUTION,
                cc.ORIENTATION,
            },
        )
        self._phase = cp.parse_unit(
            val=cp.select(
                src=config,
                key=cc.PHASE,
                default=cc.DEFAULT_PHASE,
                required=False,
            )
        )
        self._frequency = cp.select(
            src=config, key=cc.FREQUENCY, default=1.0, required=False
        )
        self._growth = cp.parse_unit(val=cp.select(src=config, key=cc.GROWTH))
        self._angle_distribution = scalar_distribution(
            config=cp.select(src=config, key=cc.ANGLE_DISTRIBUTION)
        )
        self._offset_distribution = vector_distribution(
            config=cp.select(src=config, key=cc.OFFSET_DISTRIBUTION)
        )
        self._orientation = cp.Orientation(
            config=cp.select(
                src=config,
                key=cc.ORIENTATION,
                default=cc.DEFAULT_ORIENTATION,
                required=False,
            )
        )

    @property
    def phase(self) -> PlainQuantity:
        """
        Property containing an angle describing the phase, or angle of the
        spiral from the base angle (which would be 0).
        """
        return self._phase

    @property
    def frequency(self) -> float:
        """
        Property defining the turns of the spiral for a fixed distance.  This
        is a unitless number.
        """
        return self._frequency

    @property
    def growth(self) -> PlainQuantity:
        """
        Property describing the distance the spiral grows for a given angle of
        the underlying angle distribution for the spiral.
        """
        return self._growth

    @property
    def angle_distribution(self) -> ScalarDistribution:
        """
        Distribution describing how to generate the random base angle
        associated with a particular random vector.
        """
        return self._angle_distribution

    @property
    def offset_distribution(self) -> VectorDistribution:
        """
        Distribution resulting in a 3D vector which provides a random offset
        from the random position on the spiral
        """
        return self._offset_distribution

    @property
    def orientation(self) -> cp.Orientation:
        """
        Orientation defining how to rotate the spiral after its points are
        generated in their canonical position.
        """
        return self._orientation

    def gen(self, count: int, units: str) -> np.ndarray:
        """
        Generate some number of random 3D vectors values shaped as as
        spiral storing them in a numpy array.

        Args:
            count: Number of random 3D vectors to generate
            units: Units to use when generating those vectors

        Return:
            numpy array with 'count' 3D vectors derived from the spiral's
            shape and adjusted with the types of units given.
        """
        theta = self._angle_distribution.gen(count=count, units="radian")
        r = self._growth.to(units).magnitude * np.exp(theta)
        theta = (self._frequency * theta) + self._phase.to("radian").magnitude

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.zeros(shape=count)

        points = np.vstack((x, y, z)).T
        offsets = self._offset_distribution.gen(count=count, units="meters")
        return np.dot(points + offsets, self._orientation.rotation())

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the vector generator arguments into a dictionary

        Return:
            Dictionary containing the parameters of the vector generator
        """
        return {
            self._dist_type.name.lower(): {
                cc.PHASE: {
                    cc.VALUE: self._phase.magnitude,
                    cc.UNITS: self._phase.units,
                },
                cc.FREQUENCY: self._frequency.to_dict(),
                cc.GROWTH: {
                    cc.VALUE: self._growth.magnitude,
                    cc.UNITS: self._growth.units,
                },
                cc.ANGLE_DISTRIBUTION: self._angle_distribution.to_dict(),
                cc.OFFSET_DISTRIBUTION: self._offset_distribution.to_dict(),
                cc.ORIENTATION: self._orientation.to_dict(),
            },
        }


def vector_distribution(config: Dict[str, Any]) -> VectorDistribution:
    """
    Factory for interpreting the given configuration dict and using its
    content to produce a specific generator class for the specified
    vector distribution.

    Args:
        config: Dictionary containing the specification of the vector
        generator

    Return:
        Vector distribution class instance used to create collections of
        vectors distributed based on the distribution parameters.
    """
    if len(config) != 1:
        raise ValueError(f"vector_distribution expected 1 key in {config}")
    key_name = list(config.keys())[0]
    if key_name.upper() not in set(x.name for x in VectorDistributionTypes):
        raise ValueError(f"Unknown VectorDistributionType {key_name.upper()}.")

    distribution_type = VectorDistributionTypes[key_name.upper()]

    if distribution_type == VectorDistributionTypes.FIXED_VECTOR:
        return FixedVector(config=config[key_name])

    if distribution_type == VectorDistributionTypes.ELLIPSOID:
        return Ellipsoid(config=config[key_name])

    if distribution_type == VectorDistributionTypes.CYLINDER:
        return Cylinder(config=config[key_name])

    if distribution_type == VectorDistributionTypes.SPIRAL:
        return Spiral(config=config[key_name])

    raise ValueError(f"Unknown VectorDistributionType {key_name.upper()}.")
