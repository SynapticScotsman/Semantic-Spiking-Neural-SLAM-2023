"""Small VSA helpers shared by the object-centric map.

Everything here is plain circular-convolution VSA on real vectors of a fixed
dimension ``d``: ``bind`` is FFT multiply, ``unbind`` is FFT divide by the
involution, ``bundle`` is a (normalised) sum.  These duplicate a few methods
on :class:`sspslam.sspspace.SSPSpace` deliberately so that the object map can
work on bare arrays without needing a space object in hand.
"""

import numpy as np

__all__ = [
    "bind",
    "unbind",
    "invert",
    "bundle",
    "unitary",
    "normalize",
    "cosine",
    "AtomVocab",
    "VectorCodebook",
]


def bind(a, b):
    """Circular convolution of ``a`` and ``b`` (row-wise for 2-D input)."""
    a = np.atleast_2d(np.asarray(a, dtype=float))
    b = np.atleast_2d(np.asarray(b, dtype=float))
    out = np.fft.ifft(np.fft.fft(a, axis=1) * np.fft.fft(b, axis=1), axis=1).real
    return out[0] if out.shape[0] == 1 else out


def invert(a):
    """Involution ``a^-1`` -- the exact inverse when ``a`` is unitary."""
    a = np.atleast_2d(np.asarray(a, dtype=float))
    out = a[:, -np.arange(a.shape[1])]
    return out[0] if out.shape[0] == 1 else out


def unbind(memory, key):
    """Approximate inverse of :func:`bind`: ``memory (*) key^-1``."""
    return bind(memory, invert(key))


def bundle(vectors, weights=None, normalise=True):
    """Superpose vectors (optionally weighted), returning a unit vector."""
    vectors = np.atleast_2d(np.asarray(vectors, dtype=float))
    if weights is None:
        total = vectors.mean(axis=0)
    else:
        weights = np.asarray(weights, dtype=float).reshape(-1, 1)
        wsum = np.sum(weights)
        total = np.sum(vectors * weights, axis=0) / (wsum if wsum != 0 else 1.0)
    return normalize(total) if normalise else total


def unitary(v):
    """Force every Fourier magnitude to 1, keeping the vector real.

    Use for *keys* -- identities, positions, view angles.  Do not use for
    appearance values: flattening the spectrum of an embedding-derived key
    is the whitening that costs retrieval SNR.
    """
    v = np.atleast_2d(np.asarray(v, dtype=float))
    fv = np.fft.fft(v, axis=1)
    fv = fv / np.maximum(np.abs(fv), 1e-12)
    out = np.fft.ifft(fv, axis=1).real
    return out[0] if out.shape[0] == 1 else out


def normalize(v, axis=-1):
    """L2-normalise along ``axis``."""
    v = np.asarray(v, dtype=float)
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), 1e-12)


def cosine(a, B):
    """Cosine similarity of vector ``a`` against vector or stack ``B``."""
    a = normalize(np.asarray(a, dtype=float).reshape(-1))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    return normalize(B, axis=1) @ a


class AtomVocab:
    """Mints random unitary atoms on demand and remembers them by name.

    Unlike :class:`sspslam.sspspace.SPSpace` the domain size does not have to
    be known up front, which is what an exploration walk needs: a new
    instance mints a new ``ID`` the moment it is first seen.
    """

    def __init__(self, dim, seed=None):
        self.dim = int(dim)
        self.rng = np.random.default_rng(seed)
        self._vectors = {}

    def __contains__(self, name):
        return name in self._vectors

    def __len__(self):
        return len(self._vectors)

    @property
    def names(self):
        return list(self._vectors.keys())

    def mint(self, name):
        """Create (or fetch) the atom for ``name``."""
        if name not in self._vectors:
            self._vectors[name] = unitary(self.rng.standard_normal(self.dim))
        return self._vectors[name]

    def __getitem__(self, name):
        if name not in self._vectors:
            raise KeyError(f"no atom named {name!r}; call mint() first")
        return self._vectors[name]

    def matrix(self, names=None):
        """Stack atoms into an ``(n, dim)`` array."""
        names = self.names if names is None else list(names)
        if not names:
            return np.zeros((0, self.dim))
        return np.stack([self._vectors[n] for n in names])

    def clean_up(self, v, names=None, top_k=1):
        """Nearest atoms to ``v`` by cosine similarity.

        Returns
        -------
        list of (name, similarity)
        """
        names = self.names if names is None else list(names)
        if not names:
            return []
        sims = cosine(v, self.matrix(names))
        order = np.argsort(sims)[::-1][:top_k]
        return [(names[i], float(sims[i])) for i in order]


class VectorCodebook:
    """A growable list of labelled vectors with cosine clean-up.

    Used for appearance keys, where the labels are ``(object_id, view_index)``
    pairs rather than symbol names.
    """

    def __init__(self, dim):
        self.dim = int(dim)
        self.labels = []
        self._rows = []

    def __len__(self):
        return len(self.labels)

    def add(self, label, vector):
        vector = np.asarray(vector, dtype=float).reshape(-1)
        if vector.size != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vector.size}")
        if label in self.labels:
            self._rows[self.labels.index(label)] = vector
        else:
            self.labels.append(label)
            self._rows.append(vector)

    def remove(self, label):
        if label in self.labels:
            i = self.labels.index(label)
            self.labels.pop(i)
            self._rows.pop(i)

    @property
    def matrix(self):
        if not self._rows:
            return np.zeros((0, self.dim))
        return np.stack(self._rows)

    def clean_up(self, v, top_k=1, restrict=None):
        """Nearest stored keys to ``v``.

        Parameters
        ----------
        restrict : callable or None
            Optional predicate on the label; only matching entries compete.
        """
        if not self.labels:
            return []
        idx = [i for i, lab in enumerate(self.labels)
               if restrict is None or restrict(lab)]
        if not idx:
            return []
        sims = cosine(v, self.matrix[idx])
        order = np.argsort(sims)[::-1][:top_k]
        return [(self.labels[idx[i]], float(sims[i])) for i in order]
