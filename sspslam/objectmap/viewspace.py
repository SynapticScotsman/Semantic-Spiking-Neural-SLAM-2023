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

``max_harmonic`` is also the **capacity** setting, which is the less obvious
half and the one that matters more in practice.  It fixes how many distinct
frequencies the code has, and a similarity kernel built from few frequencies
has sidelobes close to its main peak; superposing K views piles K sets of
those sidelobes together until the peak is no longer the right one.  More
frequencies, more room to superpose.

The two pull opposite ways -- more harmonics buys capacity and costs reach --
so there is an optimum, and FINDINGS.md sec.16 E2 measures it at ``4`` for a
HOG front end on a turntable, with a 28 degree lobe *wider* than the
descriptor's own 18 degree correlation length.  The default of 8 here is kept
so that everything measured before that section still reproduces; it is not
the recommended value.
"""

import numpy as np

from ..sspspace import SSPSpace, conjsym
from .geometry import wrap_angle

__all__ = ["CircularSSPSpace", "make_view_space", "residue_harmonics"]


def residue_harmonics(moduli, n_phases, weights=None):
    r"""Harmonic set for a **residue number system** over the view circle.

    Kymn et al. (NeurIPS 2024, FINDINGS.md sec.15) represent position by its
    residues in a co-prime modular system, one vector per module, and map the
    modules onto grid modules.  On a circle that construction is a *choice of
    harmonics*, which this space already supports, so no new machinery is
    needed.

    Write ``M = prod(moduli)`` and index the circle by ``x`` in ``[0, M)``
    with ``phi = 2*pi*x/M``.  The residue ``x mod m`` is carried by the phasor
    ``exp(2*pi*i*(x mod m)/m) = exp(i*(M/m)*phi)``.  So **module ``m`` is
    exactly the single integer harmonic ``M/m``** ::

        moduli (7, 8, 9)  ->  M = 504  ->  harmonics {72, 63, 56}

    and the Chinese remainder theorem is what makes those three jointly
    unambiguous over the whole circle, even though each alone repeats dozens
    of times.

    The point is where the dimensions go.  A dense band ``{1..16}`` spreads
    the budget over sixteen frequencies; the set above spends it on three,
    each with five times the redundancy, while resolving *finer* than the
    band does -- its top harmonic is 72 against 16.  That is the capacity
    claim, and sec.16 E2 measures whether it survives contact with a bundle
    of correlated appearance keys.

    The cost is the kernel shape.  A mean of three cosines has a unique global
    peak, but its sidelobes sit far higher than a tapered band's, so the
    failure mode is a jump to a wrong CRT-consistent angle rather than a
    gradual loss of precision.  Expect worse tails, not worse medians.

    Parameters
    ----------
    moduli : sequence of int
        Should be pairwise co-prime; otherwise the effective period is
        ``lcm(moduli)`` rather than the product and the code repeats.
    n_phases : int
        Slots to fill, i.e. ``(ssp_dim - 1) // 2``.
    weights : sequence of float or None
        Relative share of the slots per module.  ``None`` splits evenly,
        which is the standard construction.

    Returns
    -------
    np.ndarray
        ``(n_phases, 1)`` integer harmonics, ready to pass as
        ``CircularSSPSpace(harmonics=...)``.
    """
    moduli = [int(m) for m in moduli]
    if len(moduli) < 2 or any(m < 2 for m in moduli):
        raise ValueError("need at least two moduli, each >= 2")
    for i, a in enumerate(moduli):
        for b in moduli[i + 1:]:
            if np.gcd(a, b) != 1:
                raise ValueError(
                    f"moduli must be pairwise co-prime; {a} and {b} share "
                    f"a factor, so the code repeats every "
                    f"{np.lcm.reduce(moduli)} steps instead of "
                    f"{int(np.prod(moduli))}"
                )
    total = int(np.prod(moduli))
    base = np.array([total // m for m in moduli], dtype=float).reshape(-1, 1)
    w = (np.ones(len(moduli)) if weights is None
         else np.asarray(weights, dtype=float))
    counts = CircularSSPSpace._allocate(w, int(n_phases))
    return np.repeat(base, counts, axis=0)


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

    def view_likelihood(self, book, key, n_per_dim=720):
        r"""Similarity of ``key`` to the book read at *every* angle at once.

        This is the view-circle twin of localising in space: there you unbind
        ``ID`` from the scene map and correlate the residue against a grid of
        ``S_allo(x)``; here you correlate an observed appearance key against
        the object file read at every viewpoint.  The peak is the direction
        you are looking from.

        Scanning the circle costs one unbind per hypothesis.  It does not
        have to: because the harmonics are **integers**, the score is a
        Fourier series in the angle with integer frequencies ::

            score(phi) = <c (*) S(phi), V>
                       = (1/d) sum_f  C_f conj(V_f) exp(i k_f . phi)

        so binning ``C_f conj(V_f)`` by harmonic and taking one inverse FFT
        gives the entire likelihood field exactly -- ~640x faster than a
        720-point scan, and identical to it to machine precision.  On a 2-D
        view sphere the same thing happens with a 2-D inverse FFT.

        Parameters
        ----------
        book : np.ndarray
            The object file, ``(1/K) sum_k c_k (*) S_view(phi_k)``.
        key : np.ndarray
            Observed appearance key.
        n_per_dim : int
            Resolution of the returned field.  Costs a zero-padded FFT, so
            fine grids are nearly free.

        Returns
        -------
        (np.ndarray, np.ndarray)
            Angles of shape ``(n,)`` for a circle or ``(n, n, ..)`` grids for
            a torus, and the matching real-valued score field.
        """
        n = int(n_per_dim)
        k = np.rint(self.phase_matrix).astype(int)          # (ssp_dim, domain_dim)
        A = (np.fft.fft(np.asarray(key, dtype=float).reshape(-1))
             * np.conj(np.fft.fft(np.asarray(book, dtype=float).reshape(-1))))
        # The angle grid starts at -pi in every dimension, which puts a
        # (-1)**k twist on each coefficient.
        A = A * (-1.0) ** np.sum(k, axis=1)

        w = np.zeros((n,) * self.domain_dim, dtype=complex)
        np.add.at(w, tuple(k[:, j] % n for j in range(self.domain_dim)), A)
        field = np.fft.ifftn(w).real * (n ** self.domain_dim) / self.ssp_dim

        axis = np.linspace(-np.pi, np.pi, n, endpoint=False)
        if self.domain_dim == 1:
            return axis, field
        return np.stack(np.meshgrid(*([axis] * self.domain_dim),
                                    indexing="ij"), axis=-1), field

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
