"""
image_feature_encoder.py
========================
Maps continuous image feature vectors (SIFT, HOG, ORB, CNN embeddings, etc.)
into the unitary Semantic Pointer (SP) space used by SSP-SLAM.

The core idea: instead of assigning each landmark a discrete random SP, we
project the landmark's *appearance* feature vector into SP space via a random
projection matrix.  By the Johnson–Lindenstrauss lemma, cosine similarity is
approximately preserved, so visually similar landmarks receive similar SPs and
visually distinct ones receive dissimilar SPs.  The SPs are then made unitary
(all Fourier components have magnitude 1) so that they can be used directly
as keys in the heteroassociative memory and in circular convolution.

Usage
-----
    from sspslam.perception import ImageFeatureEncoder

    enc = ImageFeatureEncoder(feat_dim=128, ssp_dim=97, seed=0)

    # features: (feat_dim,) or (n, feat_dim) — raw descriptor from any source
    sp = enc.encode(features)           # → (ssp_dim,) or (n, ssp_dim)

    # To get the random projection matrix (e.g. to inspect or save)
    W = enc.projection                  # shape (ssp_dim, feat_dim)

Classical feature extraction helpers
-------------------------------------
If OpenCV is installed, you can use the built-in helpers:

    from sspslam.perception import extract_sift, extract_hog

    descriptor = extract_sift(image)    # numpy uint8 HxW or HxWx3
    descriptor = extract_hog(image)

These return a 1-D float32 array that can be passed directly to
``ImageFeatureEncoder.encode``.
"""

import numpy as np


class ImageFeatureEncoder:
    """Maps continuous feature vectors to unitary Semantic Pointers.

    Parameters
    ----------
    feat_dim : int
        Dimensionality of the input feature vectors (e.g. 128 for SIFT,
        variable for HOG, 512/2048 for CNN embeddings).
    ssp_dim : int
        Target dimensionality — must match the ``ssp_dim`` of the
        ``SSPSpace`` used in the SLAM network.
    seed : int or None
        Seed for the random projection matrix.  Fix this to ensure the same
        mapping is used across runs.
    normalize_input : bool
        If True (default) each input feature vector is L2-normalised before
        projection.  This makes the output invariant to descriptor magnitude
        and is appropriate for most classical feature types.
    """

    def __init__(self, feat_dim: int, ssp_dim: int, seed=None,
                 normalize_input: bool = True):
        self.feat_dim = int(feat_dim)
        self.ssp_dim = int(ssp_dim)
        self.normalize_input = normalize_input

        rng = np.random.RandomState(seed)

        # Build a (ssp_dim × feat_dim) random projection matrix.
        # Row-normalise so every projected direction has the same scale.
        W = rng.randn(self.ssp_dim, self.feat_dim)
        W = W / np.linalg.norm(W, axis=1, keepdims=True)
        self.projection = W  # (ssp_dim, feat_dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, features: np.ndarray) -> np.ndarray:
        """Project feature vector(s) to unitary SPs.

        Parameters
        ----------
        features : np.ndarray
            Shape ``(feat_dim,)`` for a single descriptor, or
            ``(n, feat_dim)`` for a batch.

        Returns
        -------
        np.ndarray
            Shape ``(ssp_dim,)`` or ``(n, ssp_dim)``.  Each row is a
            real-valued, unitary SP (all Fourier-component magnitudes = 1).
        """
        features = np.atleast_2d(features).astype(float)
        if features.shape[1] != self.feat_dim:
            raise ValueError(
                f"Expected feat_dim={self.feat_dim}, got {features.shape[1]}"
            )

        if self.normalize_input:
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            features = features / np.maximum(norms, 1e-8)

        # Linear projection: (n, feat_dim) @ (feat_dim, ssp_dim) → (n, ssp_dim)
        sp = features @ self.projection.T

        # Make unitary: normalise magnitude of every Fourier component to 1
        sp = self._make_unitary(sp)

        return sp.squeeze()

    def encode_sequence(self, feature_sequence: np.ndarray) -> np.ndarray:
        """Encode a time-series of per-landmark feature vectors.

        Parameters
        ----------
        feature_sequence : np.ndarray
            Shape ``(T, N_landmarks, feat_dim)``.

        Returns
        -------
        np.ndarray
            Shape ``(T, N_landmarks, ssp_dim)``.
        """
        T, N, F = feature_sequence.shape
        if F != self.feat_dim:
            raise ValueError(
                f"Expected feat_dim={self.feat_dim}, got {F}"
            )
        flat = feature_sequence.reshape(T * N, F)
        encoded = self.encode(flat)  # (T*N, ssp_dim)
        return encoded.reshape(T, N, self.ssp_dim)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_unitary(v: np.ndarray) -> np.ndarray:
        """Normalise Fourier-component magnitudes to 1 (keeps v real)."""
        fv = np.fft.fft(v, axis=1)
        mag = np.sqrt(fv.real ** 2 + fv.imag ** 2)
        fv = fv / np.maximum(mag, 1e-12)
        return np.fft.ifft(fv, axis=1).real


# ---------------------------------------------------------------------------
# Optional classical feature extraction helpers (require OpenCV)
# ---------------------------------------------------------------------------

def extract_sift(image: np.ndarray, n_octave_layers: int = 3) -> np.ndarray:
    """Extract a mean-pooled SIFT descriptor from an image.

    Requires ``opencv-python`` (``import cv2``).

    Parameters
    ----------
    image : np.ndarray
        Greyscale ``uint8`` array (H×W) or colour ``uint8`` (H×W×3).
        Colour images are converted to greyscale internally.

    Returns
    -------
    np.ndarray
        128-D float32 descriptor vector.  Returns a zero vector if no
        keypoints are detected.
    """
    try:
        import cv2
    except ImportError as e:
        raise ImportError(
            "extract_sift requires opencv-python: pip install opencv-python"
        ) from e

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    sift = cv2.SIFT_create(nOctaveLayers=n_octave_layers)
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None or len(descriptors) == 0:
        return np.zeros(128, dtype=np.float32)

    # Average-pool all per-keypoint descriptors → single 128-D vector
    return descriptors.mean(axis=0).astype(np.float32)


def extract_hog(image: np.ndarray,
                orientations: int = 9,
                pixels_per_cell: tuple = (8, 8),
                cells_per_block: tuple = (2, 2)) -> np.ndarray:
    """Extract a HOG feature vector from an image.

    Requires ``scikit-image`` (``from skimage.feature import hog``).

    Parameters
    ----------
    image : np.ndarray
        Greyscale or colour ``uint8`` or ``float`` array.

    Returns
    -------
    np.ndarray
        1-D float32 HOG descriptor.  Length depends on image size and
        the cell/block parameters.
    """
    try:
        from skimage.feature import hog as sk_hog
        from skimage.color import rgb2gray
    except ImportError as e:
        raise ImportError(
            "extract_hog requires scikit-image: pip install scikit-image"
        ) from e

    if image.ndim == 3:
        image = rgb2gray(image)

    descriptor = sk_hog(
        image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        feature_vector=True,
    )
    return descriptor.astype(np.float32)


def extract_orb(image: np.ndarray, n_features: int = 500) -> np.ndarray:
    """Extract a mean-pooled ORB descriptor from an image.

    Requires ``opencv-python``.

    Parameters
    ----------
    image : np.ndarray
        Greyscale ``uint8`` (H×W) or colour ``uint8`` (H×W×3).

    Returns
    -------
    np.ndarray
        32-D float32 descriptor vector (each ORB descriptor is 32 bytes).
        Returns a zero vector if no keypoints are detected.
    """
    try:
        import cv2
    except ImportError as e:
        raise ImportError(
            "extract_orb requires opencv-python: pip install opencv-python"
        ) from e

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if descriptors is None or len(descriptors) == 0:
        return np.zeros(32, dtype=np.float32)

    return descriptors.mean(axis=0).astype(np.float32)
