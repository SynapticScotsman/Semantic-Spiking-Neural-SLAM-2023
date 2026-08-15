"""Circular fractional-power encoding for the view manifold.

An ordinary :class:`~sspslam.sspspace.SSPSpace` uses real-valued phases, so
``S(x)`` never repeats -- correct for a room, wrong for an orbit.  A viewpoint
angle is periodic: walking a full circle around a chair must bring you back to
the *same* code, not a nearby one.

Restricting the phase matrix to **integer harmonics** buys exactly that::

    S(phi) = ifft( exp(i * k * phi) ),  k integer
    S(phi + 2*pi) = S(phi)                       exact, not approximate
    S(a) (*) S(b) = S(a + b  mod 2*pi)            binding is rotation

so ``bind(view_code, S_view(delta))`` orbits the object by ``delta`` with no
new image, and the code wraps cleanly at the seam.

The width of the similarity lobe is set by the largest harmonic, not by a
length scale: ``max_harmonic = M`` gives a main lobe roughly ``2*pi/M`` wide.
Small ``M`` means a view generalises over a wide arc (few, broad sides);
large ``M`` means the code distinguishes fine changes of angle but needs
more stored views to cover the circle.
"""

import numpy as np

from ..sspspace import SSPSpace, conjsym
from .geometry import wrap_angle

__all__ = ["CircularSSPSpace", "make_view_space"]


class CircularSSPSpace(SSPSpace):
    r"""Periodic FPE over one or more angles.

    Parameters
    ----------
    domain_dim : int
        ``1`` for an azimuth circle (the orbit around an object), ``2`` for
        ``(azimuth, elevation)``.  The 2-D space is a torus; keeping
        elevation inside ``[-pi/2, pi/2]`` makes it behave as a view sphere.
    ssp_dim : int
        Vector dimension.  Must be odd, and should match the dimension of the
        allocentric space so the two can be bound together.
    max_harmonic : int
        Largest integer frequency used.  Controls the angular width of the
        similarity kernel (see module docstring).
    harmonics : np.ndarray or None
        Explicit ``(n, domain_dim)`` integer phase array, overriding the
        generated one.  Entries must be integers or periodicity is lost.
    taper : {'fejer', 'flat'}
        How the dimension budget is shared between harmonics.  ``'fejer'``
        (default) trades a slightly wider main lobe for roughly half the
        sidelobe ringing; ``'flat'`` gives the sharper, noisier Dirichlet
        kernel.  See :meth:`lobe_width`.
    rng : np.random.Generator

    Attributes
    ----------
    harmonics : np.ndarray
        The ``((ssp_dim - 1) // 2, domain_dim)`` integer frequencies in use.

    Examples
    --------
    >>> space = CircularSSPSpace(1, ssp_dim=151, max_harmonic=8)
    >>> a = space.encode(0.4)
    >>> b = space.encode(0.4 + 2 * np.pi)
    >>> bool(np.allclose(a, b))
    True
    """

    def __init__(self, domain_dim=1, ssp_dim=151, max_harmonic=8,
                 harmonics=None, taper="fejer", rng=None):
        ssp_dim = int(ssp_dim)
        if ssp_dim % 2 == 0:
            raise ValueError(
                "CircularSSPSpace needs an odd ssp_dim so the harmonics can be "
                f"conjugate-symmetric; got {ssp_dim}"
            )
        rng = np.random.default_rng() if rng is None else rng
        n_phases = (ssp_dim - 1) // 2

        if harmonics is None:
            harmonics = self._make_harmonics(domain_dim, n_phases,
                                             int(max_harmonic), taper=taper)
        else:
            harmonics = np.atleast_2d(np.asarray(harmonics))
            if harmonics.shape != (n_phases, domain_dim):
                raise ValueError(
                    f"harmonics must have shape {(n_phases, domain_dim)}, "
                    f"got {harmonics.shape}"
                )
            if not np.allclose(harmonics, np.round(harmonics)):
                raise ValueError(
                    "harmonics must be integers -- non-integer frequencies "
                    "break the 2*pi periodicity that makes this a circle"
                )
            harmonics = np.round(harmonics)

        self.harmonics = harmonics.astype(float)
        self.max_harmonic = int(np.max(np.abs(self.harmonics)))

        phase_matrix = conjsym(self.harmonics)
        super().__init__(domain_dim, phase_matrix.shape[0],
                         phase_matrix=phase_matrix,
                         domain_bounds=np.tile([[-np.pi, np.pi]],
                                               (domain_dim, 1)),
                         length_scale=1, rng=rng)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @staticmethod
    def _make_harmonics(domain_dim, n_phases, max_harmonic, taper="fejer"):
        """Choose which integer frequency goes in each of the ``n_phases`` slots.

        There are far more slots than distinct harmonics below
        ``max_harmonic``, so each harmonic is repeated.  How often decides the
        shape of the similarity kernel.  Repeating them equally gives a
        Dirichlet kernel, which rings hard: at ``ssp_dim=151,
        max_harmonic=8`` its worst sidelobe is about ``-0.31``, so a view
        code can be strongly *anti*-similar to a view a little way round the
        circle.  Tapering the multiplicities linearly towards the top
        harmonic (Fejer weights) drops that to about ``-0.13`` and widens the
        main lobe from 13 to 16 degrees half-width.

        It cannot be made non-negative: exactly one dimension carries DC, so
        the kernel averages to ``1/ssp_dim`` over the circle while peaking at
        1, and must dip below zero somewhere.  The taper controls how much,
        not whether.
        """
        if max_harmonic < 1:
            raise ValueError("max_harmonic must be >= 1")
        if domain_dim == 1:
            base = np.arange(1, max_harmonic + 1, dtype=float).reshape(-1, 1)
        else:
            axes = np.arange(-max_harmonic, max_harmonic + 1, dtype=float)
            grid = np.stack(np.meshgrid(*([axes] * domain_dim), indexing="ij"),
                            axis=-1).reshape(-1, domain_dim)
            # Drop the DC term and one of every +/- pair: conjsym adds the
            # conjugate half back, so keeping both would waste dimensions.
            keep, seen = [], set()
            for row in grid:
                if not np.any(row) or tuple(-row) in seen:
                    continue
                seen.add(tuple(row))
                keep.append(row)
            base = np.stack(keep)

        if taper == "fejer":
            weights = np.prod(
                np.maximum(max_harmonic + 1 - np.abs(base), 0.0), axis=1)
        elif taper == "flat":
            weights = np.ones(base.shape[0])
        else:
            raise ValueError("taper must be 'fejer' or 'flat'")

        counts = CircularSSPSpace._allocate(weights, n_phases)
        return np.repeat(base, counts, axis=0)

    @staticmethod
    def _allocate(weights, total):
        """Integer slot counts proportional to ``weights``, summing to ``total``.

        Largest-remainder apportionment.  Every harmonic gets at least one
        slot so the top of the band is never dropped entirely; if there are
        fewer slots than harmonics -- which happens on the 2-D view sphere,
        where the harmonic grid is large -- the heaviest-weighted (lowest
        frequency) ones win.
        """
        weights = np.asarray(weights, dtype=float)
        n = weights.size
        if total < n:
            keep = np.argsort(weights)[::-1][:total]
            counts = np.zeros(n, dtype=int)
            counts[keep] = 1
            return counts
        share = weights / weights.sum() * (total - n)
        counts = np.floor(share).astype(int) + 1
        remainder = total - counts.sum()
        if remainder > 0:
            order = np.argsort(share - np.floor(share))[::-1]
            counts[order[:remainder]] += 1
        return counts

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, phi):
        """Encode angle(s), wrapping into ``[-pi, pi)`` first."""
        phi = np.atleast_2d(np.asarray(phi, dtype=float))
        if phi.shape[-1] != self.domain_dim:
            phi = phi.reshape(-1, self.domain_dim)
        out = super().encode(wrap_angle(phi))
        return out[0] if out.shape[0] == 1 else out

    def update_lengthscale(self, scale):  # pragma: no cover - guard rail
        raise NotImplementedError(
            "a view circle has no length scale; set max_harmonic instead "
            "(scaling the phases would make the code aperiodic)"
        )

    # ------------------------------------------------------------------
    # Read-out
    # ------------------------------------------------------------------

    def similarity_kernel(self, delta):
        """``S(0) . S(delta)`` in closed form, for choosing ``max_harmonic``."""
        delta = np.atleast_2d(np.asarray(delta, dtype=float))
        if delta.shape[-1] != self.domain_dim:
            delta = delta.reshape(-1, self.domain_dim)
        phases = delta @ self.harmonics.T           # (n_delta, n_harmonics)
        sims = (1 + 2 * np.sum(np.cos(phases), axis=1)) / self.ssp_dim
        return sims[0] if sims.size == 1 else sims

    def lobe_width(self, level=0.5, n=721):
        """Half-width of the main similarity lobe, in radians.

        The angular distance at which the kernel first falls to ``level``
        times its peak.  A sensible scale for how far apart stored views
        should be, and for the merge tolerance when writing them.
        """
        deltas = np.linspace(0, np.pi, n)
        probe = np.zeros((n, self.domain_dim))
        probe[:, 0] = deltas
        sims = np.atleast_1d(self.similarity_kernel(probe))
        below = np.nonzero(sims < level * sims[0])[0]
        return float(deltas[below[0]]) if below.size else np.pi

    def sample_grid(self, n_per_dim=181):
        """Angles and their codes on a regular grid over the circle/torus."""
        axis = np.linspace(-np.pi, np.pi, n_per_dim, endpoint=False)
        if self.domain_dim == 1:
            pts = axis.reshape(-1, 1)
        else:
            pts = np.stack(
                np.meshgrid(*([axis] * self.domain_dim), indexing="ij"),
                axis=-1).reshape(-1, self.domain_dim)
        return pts, np.atleast_2d(self.encode(pts))

    def decode(self, ssp, n_per_dim=361, **kwargs):
        """Angle whose code best matches ``ssp`` (grid search over the circle).

        Overrides the base-class decoder, which assumes a non-periodic
        domain and a length scale.
        """
        pts, codes = self.sample_grid(n_per_dim)
        ssp = np.asarray(ssp, dtype=float).reshape(-1)
        sims = codes @ ssp
        best = int(np.argmax(sims))
        return pts[best] if self.domain_dim > 1 else float(pts[best, 0])


def make_view_space(ssp_dim, view_dims=1, max_harmonic=8, rng=None):
    """Convenience constructor matching an existing allocentric dimension."""
    return CircularSSPSpace(domain_dim=view_dims, ssp_dim=ssp_dim,
                            max_harmonic=max_harmonic, rng=rng)
