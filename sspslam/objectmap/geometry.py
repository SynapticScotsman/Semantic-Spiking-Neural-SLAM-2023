"""Angle bookkeeping for object-centred view codes.

Two different angles show up when a robot looks at an object, and mixing them
up silently corrupts the view book, so they get separate names here.

``view_azimuth``
    *Which side of the object am I looking at?*  Measured at the object,
    from the object's own front direction, towards the robot.  This is the
    coordinate the view circle ``S_view`` is an FPE of.

``camera_bearing``
    *Where in my field of view is it?*  Measured at the robot, from the
    robot's heading, towards the object.  This is what you use to point a
    camera or a saccade; it is **not** a property of the object and must not
    be stored in the object file.

The two are related by ``view_azimuth = camera_bearing + theta_robot
- yaw_obj + pi`` (mod 2*pi), i.e. they differ by a half turn plus the
object's own orientation.  If an object has no meaningful front (a bin, a
pillar), pick ``yaw_obj = 0`` once and keep it fixed -- the view circle is
then anchored to the world frame instead of the object frame, which is fine
as long as the object does not rotate.
"""

import numpy as np

__all__ = [
    "wrap_angle",
    "view_azimuth",
    "camera_bearing",
    "view_azimuth_elevation",
    "circular_mean",
    "angular_distance",
]


def wrap_angle(theta):
    """Wrap angle(s) into ``[-pi, pi)``."""
    return (np.asarray(theta, dtype=float) + np.pi) % (2 * np.pi) - np.pi


def angular_distance(a, b):
    """Smallest signed difference ``a - b`` wrapped into ``[-pi, pi)``."""
    return wrap_angle(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))


def view_azimuth(obj_pos, robot_pos, obj_yaw=0.0):
    """Object-centred azimuth of the viewer -- the side of the object seen.

    Parameters
    ----------
    obj_pos : array_like
        World position of the object, ``(2,)`` or ``(3,)`` (only x, y used).
    robot_pos : array_like
        World position of the camera, same shape.
    obj_yaw : float
        Heading of the object's own front in the world frame.  ``0`` anchors
        the view circle to the world x axis.

    Returns
    -------
    float
        Azimuth in ``[-pi, pi)``.  ``0`` means the camera is on the object's
        front side, ``pi`` means directly behind it.
    """
    obj_pos = np.asarray(obj_pos, dtype=float).reshape(-1)
    robot_pos = np.asarray(robot_pos, dtype=float).reshape(-1)
    d = robot_pos[:2] - obj_pos[:2]
    return float(wrap_angle(np.arctan2(d[1], d[0]) - obj_yaw))


def camera_bearing(obj_pos, robot_pos, robot_yaw=0.0):
    """Bearing of the object in the robot's own frame -- where to look.

    This is the *query-time* quantity: never store it in an object file.
    """
    obj_pos = np.asarray(obj_pos, dtype=float).reshape(-1)
    robot_pos = np.asarray(robot_pos, dtype=float).reshape(-1)
    d = obj_pos[:2] - robot_pos[:2]
    return float(wrap_angle(np.arctan2(d[1], d[0]) - robot_yaw))


def view_azimuth_elevation(obj_pos, robot_pos, obj_yaw=0.0):
    """Object-centred ``(azimuth, elevation)`` -- the sphere version.

    Elevation is measured from the object's horizontal plane, positive when
    the camera is above the object.  Returned in ``[-pi/2, pi/2]``; the
    2-D view space that consumes it is a torus, so keep elevation inside
    that half-turn band and it behaves like a sphere.
    """
    obj_pos = np.asarray(obj_pos, dtype=float).reshape(-1)
    robot_pos = np.asarray(robot_pos, dtype=float).reshape(-1)
    if obj_pos.size < 3 or robot_pos.size < 3:
        raise ValueError("view_azimuth_elevation needs 3-D positions")
    d = robot_pos - obj_pos
    az = wrap_angle(np.arctan2(d[1], d[0]) - obj_yaw)
    el = np.arctan2(d[2], np.linalg.norm(d[:2]))
    return float(az), float(el)


def circular_mean(angles, weights=None):
    """Weighted circular mean of angles, in ``[-pi, pi)``."""
    angles = np.asarray(angles, dtype=float).reshape(-1)
    if weights is None:
        weights = np.ones_like(angles)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    s = np.sum(weights * np.sin(angles))
    c = np.sum(weights * np.cos(angles))
    return float(wrap_angle(np.arctan2(s, c)))
