"""Object-centric VSA cognitive map.

A scene map says which objects exist and where; an object file says what each
instance looks like from which side.  See :mod:`.scenemap` for the split and
why the two memories are not merged into one blob.

Only numpy and scipy are needed here -- no nengo, no vision packages.
"""

from .appearance import AppearanceCodec
from .geometry import (angular_distance, camera_bearing, circular_mean,
                       view_azimuth, view_azimuth_elevation, wrap_angle)
from .objectfile import ObjectFile, ViewEntry
from .scenemap import ObjectCentricMap, PlaceQuery, ViewQuery
from .viewspace import CircularSSPSpace, make_view_space
from .vsa import (AtomVocab, VectorCodebook, bind, bundle, cosine, invert,
                  normalize, unbind, unitary)

__all__ = [
    "ObjectCentricMap",
    "ObjectFile",
    "ViewEntry",
    "PlaceQuery",
    "ViewQuery",
    "CircularSSPSpace",
    "make_view_space",
    "AppearanceCodec",
    "AtomVocab",
    "VectorCodebook",
    "bind",
    "unbind",
    "invert",
    "bundle",
    "unitary",
    "normalize",
    "cosine",
    "wrap_angle",
    "angular_distance",
    "circular_mean",
    "view_azimuth",
    "camera_bearing",
    "view_azimuth_elevation",
]
