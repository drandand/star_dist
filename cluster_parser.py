"""
cluster_parser.py

Collection of routines which parse different types of dict
entries.
"""

from typing import Any, Dict
from pint.facets.plain import PlainQuantity
import pint
import pint.facets
import numpy as np
import cluster_common as cc


def select(
    src: dict,
    key: str,
    default: Any = None,
    required: bool = True,
) -> Any:
    """
    Select an element from a dictionary with the named key.  If the required
    flag is false and the key element is not available, then use the default
    value.  If required is true and the key is not in the dict, then raise a
    KeyError.

    Args:
        src - Dictionary potentially containing the key
        key - Key in the dictionary to retrieve
        default - Default value to use if the required flag is false and the
        key is not in the dict
        required - Flag to indicate whether the key is required

    Return:
        Return the dictionary element with the given key or te default if the
        key is absent and required flag is false.

    Raises:
        KeyError if the key is missing and the required flag is True
    """
    if required and key not in src:
        raise KeyError(f"Required key {key} not found in {src=}")

    return src[key] if key in src else default


def parse_unit(val: Dict[str, float | str]) -> PlainQuantity[Any]:
    """
    Parse a dictionary to retrieve a value and it associated units

    Args:
        val - Dictionary containing the value and units elements

    Return
        pint quantity / unit comprised of the elements passed
    """
    cc.validate(config=val, keys={cc.VALUE, cc.UNITS})
    value: float = float(select(src=val, key=cc.VALUE))
    units: str = str(select(src=val, key=cc.UNITS))
    ret = value * cc.ureg(units)
    return ret


def to_dict(quantity: pint.facets.plain.PlainQuantity[Any]) -> Dict[str, Any]:
    """
    Convert the quantity to a dictionary with value and unit tags

    Args:
      quantity - to convert to a dictionary

    Return:
        A dictionary representation of the given quantity
    """
    return {cc.VALUE: quantity.m, cc.UNITS: str(quantity.u)}


class UnitsVector3d:
    """
    A 3-D vector with units

    Args:
      config - Dictionary containing the elements to populate corresponding
      vector elements
    """

    def __init__(self, config: dict):
        cc.validate(config=config, keys={cc.X, cc.Y, cc.Z})
        self._x = parse_unit(select(src=config, key=cc.X))
        self._y = parse_unit(select(src=config, key=cc.Y))
        self._z = parse_unit(select(src=config, key=cc.Z))

    @property
    def x(self) -> pint.facets.plain.PlainQuantity:
        """Accessor for the X element of the vector"""
        return self._x

    @property
    def y(self) -> pint.facets.plain.PlainQuantity:
        """Accessor for the Y element of the vector"""
        return self._y

    @property
    def z(self) -> pint.facets.plain.PlainQuantity:
        """Accessor for the Z element of the vector"""
        return self._z

    def to(self, units: str) -> "UnitsVector3d":
        """
        Return a UnitsVector3d which is a copy of this one with the units
        updated based on the argument given.

        Args:
          units - Units to convert vector element

        Return:
            A new vector based on the given units derived from this vector
        """
        return UnitsVector3d(
            {
                cc.X: {cc.VALUE: self._x.to(units).magnitude, cc.UNITS: units},
                cc.Y: {cc.VALUE: self._y.to(units).magnitude, cc.UNITS: units},
                cc.Z: {cc.VALUE: self._z.to(units).magnitude, cc.UNITS: units},
            }
        )

    def to_numpy(self, units: str | None = None) -> np.ndarray:
        """
        Convert this vector into a numpy vector converted to the units
        provided (if any)

        Args:
            units - Units used to represent the elements of the resulting
            numpy vector

        Return:
            A numpy vector equivalent to this converted to the given units
        """
        return (
            np.array([self._x.magnitude, self._y.magnitude, self._z.magnitude])
            if units is None
            else self.to(units=units).to_numpy()
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Units Vector into a dictionary

        Return:
            A dictionary representation of this vector
        """
        return {
            cc.X: to_dict(quantity=self._x),
            cc.Y: to_dict(quantity=self._y),
            cc.Z: to_dict(quantity=self._z),
        }


class Vector3d:
    """
    A 3-D vector without units

    Args:
      config - Dictionary containing the elements to populate corresponding
      vector elements
    """

    def __init__(self, config: dict):
        cc.validate(config=config, keys={cc.X, cc.Y, cc.Z})
        self._x: float = float(select(src=config, key=cc.X))
        self._y: float = float(select(src=config, key=cc.Y))
        self._z: float = float(select(src=config, key=cc.Z))

    @property
    def x(self) -> float:
        """Accessor for the X element of the vector"""
        return self._x

    @property
    def y(self) -> float:
        """Accessor for the Y element of the vector"""
        return self._y

    @property
    def z(self) -> float:
        """Accessor for the Z element of the vector"""
        return self._z

    def to_numpy(self) -> np.ndarray:
        """Convert the vector into a numpy array"""
        return np.array([self._x, self._y, self._z])

    def to_dict(self) -> Dict[str, Any]:
        """Convert the vector into a dictionary"""
        return {cc.X: self._x, cc.Y: self._y, cc.Z: self._z}


class Ellipsoid:
    """
    Contains the parameters for describing an ellipsoid

    Args:
        config - Dictionary containing the unitless values describing the
        shape of the ellipsoid.
    """

    def __init__(self, config: dict):
        cc.validate(config=config, keys={cc.A, cc.B, cc.C})
        self._a: float = float(select(src=config, key=cc.A))
        self._b: float = float(select(src=config, key=cc.B))
        self._c: float = float(select(src=config, key=cc.C))

    @property
    def a(self) -> float:
        """Accessor for the a parameter of the ellipsoid"""
        return self._a

    @property
    def b(self) -> float:
        """Accessor for the b parameter of the ellipsoid"""
        return self._b

    @property
    def c(self) -> float:
        """Accessor for the c parameter of the ellipsoid"""
        return self._c

    def to_dict(self) -> Dict[str, Any]:
        """Convert the ellipsoid object into a dictionary"""
        return {cc.A: self._a, cc.B: self._b, cc.C: self._c}


class Orientation:
    """
    Class containing the arguments needed to define an orientation which
    consists of an axis and angle of a rotation.  The vector is three
    dimensional and of non-zero length.  The angle specifies units.

    Args:
        config - Dict containing the elements of the orientation
    """

    def __init__(self, config: dict):
        cc.validate(config=config, keys={cc.AXIS, cc.THETA})
        self._axis = Vector3d(config=select(src=config, key=cc.AXIS))
        self._theta = parse_unit(select(src=config, key=cc.THETA))

    @property
    def axis(self) -> Vector3d:
        """Accessor for the axis (a 3d vector) of rotation"""
        return self._axis

    @property
    def theta(self) -> pint.facets.plain.PlainQuantity:
        """Accessor for the angle with units"""
        return self._theta

    def rotation(self) -> np.ndarray:
        """
        Generate a rotation matrix as a numpy array derived from
        the rotation axis and angle
        """
        v = self._axis.to_numpy()
        v = v / np.linalg.norm(v)
        v_x, v_y, v_z = v
        theta = self._theta.to("radian").magnitude

        kernel = np.array([[0, -v_z, v_y], [v_z, 0, -v_x], [-v_y, v_x, 0]])

        identity = np.eye(3)
        rot = (
            identity
            + np.sin(theta) * kernel
            + (1 - np.cos(theta)) * np.dot(kernel, kernel)
        )

        return rot.T

    def to(self, units: str) -> "Orientation":
        """
        Convert the orientation instance from the current units to the units
        given.

        Args:
            units - String name of the angle units to convert the rotation
            angle to

        Return:
            A new orientation using the new units for the angle
        """
        return Orientation(
            config={
                cc.AXIS: self._axis.to_dict(),
                cc.THETA: {
                    cc.VALUE: self._theta.to(units).magnitude,
                    cc.UNITS: units,
                },
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this orientation to a dictionary

        Return:
            Dictionary representation of the orientation
        """
        return {
            cc.AXIS: self._axis.to_dict(),
            cc.THETA: to_dict(quantity=self._theta),
        }


class Render:
    """
    Class defining how to render components, which comprises of the elements of
    a color and a radius for a circular depiction.

    Args:
        config - Dictionary containing the elements to use to define the
        render class instance
    """

    def __init__(self, config: dict):
        cc.validate(config=config, keys={cc.R, cc.G, cc.B, cc.RADIUS})
        self._r: float = float(select(src=config, key=cc.R))
        self._g: float = float(select(src=config, key=cc.G))
        self._b: float = float(select(src=config, key=cc.B))
        self._radius: float = float(select(src=config, key=cc.RADIUS))

    @property
    def r(self) -> float:
        """Accessor for the red component of the render object"""
        return self._r

    @property
    def g(self) -> float:
        """Accessor for the green component of the render object"""
        return self._g

    @property
    def b(self) -> float:
        """Accessor for the blue component of the render object"""
        return self._b

    @property
    def radius(self) -> float:
        """Accessor for the radius component of the render object"""
        return self._radius

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the render object into a dictionary

        Return:
            Dictionary representation of the render object
        """
        return {
            cc.R: self._r,
            cc.G: self._g,
            cc.B: self._b,
            cc.RADIUS: self._radius,
        }
