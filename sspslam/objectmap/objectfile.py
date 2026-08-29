"""The per-instance object file: what this thing looks like from which side.

An object file is a small view-book, not an image store::

    V_obj = (1/K) * sum_k  c(z_k) (*) S_view(phi_k)

``c(z_k)`` is an appearance key (see :mod:`.appearance`) and ``S_view(phi_k)``
is a periodic FPE of the object-centred azimuth (see :mod:`.viewspace`).  The
angle is the index; the appearance is the value stored at that index.

``K`` stays small on purpose.  Every frame of a walk past a chair is nearly
the same view, and bundling a hundred near-duplicates costs capacity without
adding information, so nearby observations merge into one entry instead of
appending.  A handful of well-separated sides is the target.

Note what is *not* in here: the robot's pose.  The viewpoint angle is a
relation between the object and wherever the camera happens to be, so it is
computed at query time from the two poses.  Storing ego snapshots would make
the file a record of places the robot stood, which stops being true the
moment the object moves.
"""

from dataclasses import dataclass, field

import numpy as np

from .geometry import angular_distance, circular_mean, wrap_angle
from .vsa import bind, bundle, normalize

__all__ = ["ViewEntry", "ObjectFile"]


@dataclass
class ViewEntry:
    """One side of an object: an angle and the appearance key seen there."""

    phi: np.ndarray
    key: np.ndarray
    count: int = 1

    def merged_with(self, phi, key):
        """Running mean of this entry with a new observation."""
        n = self.count
        new_phi = np.array([
            circular_mean([self.phi[i], wrap_angle(np.atleast_1d(phi)[i])],
                          weights=[n, 1])
            for i in range(self.phi.size)
        ])
        new_key = normalize((self.key * n + np.asarray(key, float)) / (n + 1))
        return ViewEntry(phi=new_phi, key=new_key, count=n + 1)


@dataclass
class ObjectFile:
    """One instance: its identity, where it is, and its view book.

    Attributes
    ----------
    obj_id : str
        Instance name.  Unique per physical thing, not per class.
    id_vec : np.ndarray
        The random unitary atom standing for this instance.  The crop is
        never used as the name -- a name has to survive the object being
        seen from a new side or under new lighting.
    class_name : str or None
        Optional category label, backed by its own atom in the map.
    position : np.ndarray
        World position (the argument of ``S_allo``).
    yaw : float
        World heading of the object's front, defining ``phi = 0``.
    views : list of ViewEntry
    n_observations : int
        Total detections folded in, including merged ones.
    history : list of (float, np.ndarray)
        ``(timestamp, position)`` whenever the position changed.
    """

    obj_id: str
    id_vec: np.ndarray
    position: np.ndarray
    yaw: float = 0.0
    class_name: str = None
    views: list = field(default_factory=list)
    n_observations: int = 0
    history: list = field(default_factory=list)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def add_view(self, phi, key, merge_tol=np.deg2rad(20.0)):
        """Add an observed side, merging into a nearby entry if there is one.

        Returns
        -------
        int
            Index of the entry that now holds this observation.
        """
        phi = wrap_angle(np.atleast_1d(np.asarray(phi, dtype=float)))
        key = normalize(np.asarray(key, dtype=float).reshape(-1))
        self.n_observations += 1

        if merge_tol is not None and self.views:
            gaps = [np.max(np.abs(angular_distance(phi, v.phi)))
                    for v in self.views]
            nearest = int(np.argmin(gaps))
            if gaps[nearest] <= merge_tol:
                self.views[nearest] = self.views[nearest].merged_with(phi, key)
                return nearest

        self.views.append(ViewEntry(phi=phi, key=key))
        return len(self.views) - 1

    def set_position(self, position, timestamp=None):
        """Move the object.  The view book is kept -- that is the point."""
        self.history.append((timestamp, np.array(self.position, dtype=float)))
        self.position = np.asarray(position, dtype=float).reshape(-1)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    @property
    def n_views(self):
        return len(self.views)

    @property
    def angles(self):
        """``(K, view_dims)`` array of the stored viewpoint angles."""
        if not self.views:
            return np.zeros((0, 1))
        return np.stack([v.phi for v in self.views])

    @property
    def keys(self):
        """``(K, dim)`` array of the stored appearance keys."""
        if not self.views:
            return np.zeros((0, self.id_vec.size))
        return np.stack([v.key for v in self.views])

    def view_book(self, view_space, weight_by_count=False):
        """``(1/K) sum_k c_k (*) S_view(phi_k)`` -- the object file itself."""
        if not self.views:
            return np.zeros(self.id_vec.size)
        terms = np.stack([bind(v.key, view_space.encode(v.phi))
                          for v in self.views])
        weights = ([v.count for v in self.views] if weight_by_count else None)
        return bundle(terms, weights=weights)

    def prototype(self, weight_by_count=False):
        """``(1/K) sum_k c_k`` -- the appearance book with no angle bound in.

        The view book answers *which way am I looking at this?*.  This answers
        *what am I looking at?*, and it is deliberately the same keys with the
        binding left out.

        Worth its own vector because the two questions have opposite
        requirements.  Binding an angle spreads each key across the whole
        code, so K views of one object interfere; FINDINGS.md sec.16 E0 shows
        the view book's identification *falling* as K grows, because the
        bundle saturates.  Bundling the keys unbound has no such pressure --
        the terms are correlated by construction (they are the same object)
        and reinforce instead of interfering, which is exactly the identity
        pedestal of sec.14 used as signal rather than fought as noise.

        Measured in sec.16 E1: at ``ssp_dim=151`` this identifies at 0.89
        against the view book's 0.44, and unlike the view book it does not
        care about dimension.  Cost is one extra vector per object.
        """
        if not self.views:
            return np.zeros(self.id_vec.size)
        weights = ([v.count for v in self.views] if weight_by_count else None)
        return bundle(self.keys, weights=weights)

    def coverage(self, view_space, n_per_dim=181):
        """Fraction of the view circle within the main lobe of a stored view.

        A crude "have I walked enough of the way around this thing" number:
        it counts grid angles whose similarity to some stored angle exceeds
        half the kernel peak.
        """
        if not self.views:
            return 0.0
        pts, _ = view_space.sample_grid(n_per_dim)
        stored = self.angles
        half_peak = 0.5 * float(view_space.similarity_kernel(
            np.zeros(view_space.domain_dim)))
        covered = np.zeros(pts.shape[0], dtype=bool)
        for phi in stored:
            sims = view_space.similarity_kernel(wrap_angle(pts - phi))
            covered |= np.atleast_1d(sims) >= half_peak
        return float(covered.mean())
