"""Appearance keys: crop embedding -> view-dependent VSA value.

The rule this module encodes is a negative one as much as a positive one.

**Do** turn a crop embedding into a key with a fixed random projection and an
L2 normalisation::

    c(z) = unit( (z - mu) W )                 optionally z-scored first

**Do not** fractional-power-encode the embedding (``exp(i W z)``).  FPE builds
a *manifold*: it makes codes for similar embeddings similar and lets you
interpolate between them.  That is the right thing for space and for
viewpoint, and the wrong thing for appearance, because "similar embedding"
means "looks alike", which merges two different chairs rather than two sides
of one chair.  The sides are already indexed by ``S_view``; appearance is the
*value* stored at each index, so it wants to stay a plain, roughly orthogonal
key.

**Do not** whiten either.  Centering (or z-scoring) removes the shared mean
direction that otherwise makes every crop look alike, which is worth doing.
Full whitening keeps going, amplifying the low-variance directions where the
embedding carries mostly noise, and retrieval SNR drops.  ``standardize`` is
therefore limited to ``'center'`` / ``'zscore'`` / ``'none'`` on purpose.

Statistics can be fitted up front from a sample of crops, or accumulated
online during a walk with :meth:`AppearanceCodec.partial_fit`.
"""

import numpy as np

from .vsa import normalize, unitary

__all__ = ["AppearanceCodec"]


class AppearanceCodec:
    """Map crop embeddings to appearance keys of the map's dimension.

    Parameters
    ----------
    feat_dim : int
        Dimension of the incoming crop embedding (e.g. 384 for DINOv2-S,
        512 for a CLIP image tower).
    ssp_dim : int
        Vector dimension of the map.
    standardize : {'center', 'zscore', 'none'}
        Per-feature statistics removed before projection.  ``'center'`` is
        the default; ``'zscore'`` also divides by the per-feature standard
        deviation.  Whitening is deliberately not offered.
    seed : int or None
        Seed of the projection matrix.  Fix it: keys minted in different
        sessions must live in the same space.
    make_unitary : bool
        Off by default.  Turning it on flattens the key's Fourier spectrum,
        which is the whitening this module warns about; it is exposed only so
        the effect can be measured.

    Attributes
    ----------
    projection : np.ndarray
        ``(feat_dim, ssp_dim)`` random projection ``W``.
    n_seen : int
        Number of embeddings folded into the running statistics.
    """

    def __init__(self, feat_dim, ssp_dim, standardize="center", seed=None,
                 make_unitary=False):
        if standardize not in ("center", "zscore", "none"):
            raise ValueError(
                "standardize must be 'center', 'zscore' or 'none'; whitening "
                "is not offered because it costs retrieval SNR"
            )
        self.feat_dim = int(feat_dim)
        self.ssp_dim = int(ssp_dim)
        self.standardize = standardize
        self.make_unitary = bool(make_unitary)

        rng = np.random.default_rng(seed)
        W = rng.standard_normal((self.feat_dim, self.ssp_dim))
        self.projection = W / np.sqrt(self.feat_dim)

        self.mean = np.zeros(self.feat_dim)
        self.var = np.ones(self.feat_dim)
        self.n_seen = 0

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def fit(self, embeddings):
        """Set the statistics from a batch of embeddings."""
        Z = np.atleast_2d(np.asarray(embeddings, dtype=float))
        self._check_dim(Z)
        self.mean = Z.mean(axis=0)
        self.var = Z.var(axis=0)
        self.n_seen = Z.shape[0]
        return self

    def partial_fit(self, embeddings):
        """Fold more embeddings into the running mean/variance (Welford)."""
        Z = np.atleast_2d(np.asarray(embeddings, dtype=float))
        self._check_dim(Z)
        for z in Z:
            self.n_seen += 1
            delta = z - self.mean
            self.mean = self.mean + delta / self.n_seen
            self.var = self.var + (delta * (z - self.mean) - self.var) / self.n_seen
        return self

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, embeddings):
        """Embedding(s) -> appearance key(s) of dimension ``ssp_dim``."""
        Z = np.atleast_2d(np.asarray(embeddings, dtype=float))
        self._check_dim(Z)
        single = Z.shape[0] == 1

        if self.standardize != "none" and self.n_seen > 0:
            Z = Z - self.mean
            if self.standardize == "zscore":
                Z = Z / np.maximum(np.sqrt(self.var), 1e-8)

        keys = normalize(Z @ self.projection, axis=1)
        if self.make_unitary:
            keys = np.atleast_2d(unitary(keys))
        return keys[0] if single else keys

    def _check_dim(self, Z):
        if Z.shape[1] != self.feat_dim:
            raise ValueError(
                f"expected feat_dim={self.feat_dim}, got {Z.shape[1]}"
            )
