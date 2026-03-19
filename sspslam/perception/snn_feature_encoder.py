"""
snn_feature_encoder.py
======================
Maps spiking neural network (SNN) latent representations into the unitary
Semantic Pointer (SP) space used by SSP-SLAM.

This module provides two public classes:

NengoSNN
    A two-layer Nengo-based spiking network (LIF neurons) that processes
    images offline and returns a decoded latent vector.  Uses the same Nengo
    framework already depended on by the rest of the project.

SNNFeatureEncoder
    Maps *any* continuous latent vector (from NengoSNN or an external SNN such
    as snnTorch, Norse, BindsNET, etc.) into a unitary SP via the same
    random-projection approach used by ``ImageFeatureEncoder``.  Its
    ``encode()`` method is duck-type-compatible with ``ImageFeatureEncoder``,
    so it can be passed directly to
    ``get_slam_input_functions_from_features()`` with no modifications to the
    SLAM network code.

Usage — external latent vectors (any SNN framework)
----------------------------------------------------
    from sspslam.perception import SNNFeatureEncoder

    # latent_dim: dimensionality produced by your SNN
    enc = SNNFeatureEncoder(latent_dim=64, ssp_dim=97, seed=0)

    # latent: (latent_dim,) or (n, latent_dim) numpy array from your SNN
    sp = enc.encode(latent)           # → (ssp_dim,) or (n, ssp_dim)

Usage — built-in Nengo two-layer SNN
--------------------------------------
    from sspslam.perception import SNNFeatureEncoder, NengoSNN

    snn = NengoSNN(
        input_dim=128 * 128,   # flattened image size
        latent_dim=64,
        n_neurons_layer1=200,
        n_neurons_layer2=100,
        seed=0,
    )
    enc = SNNFeatureEncoder(latent_dim=64, ssp_dim=97, seed=0, nengo_snn=snn)

    sp = enc.extract_and_encode(image)   # numpy uint8 HxW or HxWx3

Integrating with the SLAM pipeline
------------------------------------
    from sspslam.networks.slam import get_slam_input_functions_from_features

    # SNNFeatureEncoder is duck-type-compatible — no other changes needed
    result = get_slam_input_functions_from_features(
        ssp_space, enc, velocity_data, vec_to_landmarks, latent_vectors, view_rad
    )
"""

import numpy as np


# ---------------------------------------------------------------------------
# NengoSNN  — two-layer Nengo spiking network
# ---------------------------------------------------------------------------

class NengoSNN:
    """Two-layer Nengo LIF spiking network that encodes images as latent vectors.

    Architecture
    ------------
    ::

        input node (input_dim,)
            │  random projection transform  (n_neurons_layer1, input_dim)
            ▼
        Layer-1 ensemble  [n_neurons_layer1 LIF neurons]
            │  decoded linear connection  (n_neurons_layer2, n_neurons_layer1) via decoders
            ▼
        Layer-2 ensemble  [n_neurons_layer2 LIF neurons]
            │  decoded output  (latent_dim,)
            ▼
        Probe → latent vector

    The first connection uses a random Gaussian projection scaled by
    ``1/sqrt(input_dim)`` so activations stay in a reasonable range regardless
    of image size.  The second connection is a learned linear decoding
    optimised by Nengo's built-in least-squares solver.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the flattened input image (e.g. 128*128 = 16 384).
        Colour images (H×W×3) should be converted to greyscale before
        flattening, or ``input_dim`` set to H*W*3.
    latent_dim : int
        Dimensionality of the decoded output latent vector.
    n_neurons_layer1 : int
        LIF neuron count in the first spiking layer.  Default 200.
    n_neurons_layer2 : int
        LIF neuron count in the second spiking layer.  Default 100.
    presentation_time : float
        Duration (seconds) each image is held constant during simulation.
        Longer values let spike rates stabilise.  Default 0.1 s.
    synapse : float
        Low-pass synaptic filter time-constant (seconds) on the output probe.
        Default 0.05 s.
    seed : int or None
        Fixes both the network build seed and the simulator seed for
        reproducible latent vectors.
    """

    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 n_neurons_layer1: int = 200,
                 n_neurons_layer2: int = 100,
                 presentation_time: float = 0.1,
                 synapse: float = 0.05,
                 seed=None):
        try:
            import nengo
        except ImportError as e:
            raise ImportError(
                "NengoSNN requires nengo: pip install nengo"
            ) from e

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.n_neurons_layer1 = int(n_neurons_layer1)
        self.n_neurons_layer2 = int(n_neurons_layer2)
        self.presentation_time = float(presentation_time)
        self.synapse = float(synapse)
        self.seed = seed

        # Pre-compute the fixed random projection for layer 1
        rng = np.random.RandomState(seed)
        self._W1 = rng.randn(n_neurons_layer1, input_dim) / np.sqrt(input_dim)

        # Build the Nengo network once; reuse across calls
        self._nengo = nengo
        self._model = None   # built lazily on first call to extract()
        self._input_signal = None  # mutable placeholder updated per image

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Run the SNN on a single image and return the latent vector.

        Parameters
        ----------
        image : np.ndarray
            Shape (H, W) or (H, W, C) uint8 or float.  Colour images are
            converted to greyscale by averaging channels.

        Returns
        -------
        np.ndarray
            Shape (latent_dim,).  Decoded output of the second spiking layer
            at the end of the presentation window.
        """
        flat = self._preprocess(image)
        return self._run_sim(flat)

    def extract_batch(self, images) -> np.ndarray:
        """Run the SNN on a sequence of images.

        Parameters
        ----------
        images : list or array of np.ndarray
            Each element has shape (H, W) or (H, W, C).

        Returns
        -------
        np.ndarray
            Shape (N, latent_dim).
        """
        return np.stack([self.extract(img) for img in images])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Flatten and normalise an image to shape (input_dim,)."""
        img = np.array(image, dtype=float)
        if img.ndim == 3:
            img = img.mean(axis=2)          # greyscale
        flat = img.flatten()
        if flat.shape[0] != self.input_dim:
            raise ValueError(
                f"Image flattened to {flat.shape[0]} elements, "
                f"but input_dim={self.input_dim}."
            )
        # L∞ normalise to [-1, 1]
        mx = np.max(np.abs(flat))
        if mx > 0:
            flat = flat / mx
        return flat.astype(float)

    def _build_model(self, flat_image: np.ndarray):
        """Construct the Nengo model (done once, then reused)."""
        nengo = self._nengo

        # Mutable list so the lambda inside the Node can see updates
        self._current_input = [flat_image.copy()]

        with nengo.Network(seed=self.seed) as model:
            # Input node: returns the current image each timestep
            inp = nengo.Node(lambda t: self._current_input[0])

            # Layer 1: random projection of image pixels → LIF ensemble
            layer1 = nengo.Ensemble(
                n_neurons=self.n_neurons_layer1,
                dimensions=self.n_neurons_layer1,
                neuron_type=nengo.LIF(),
                seed=self.seed,
            )
            nengo.Connection(
                inp, layer1.neurons,
                transform=self._W1,
                synapse=None,
            )

            # Layer 2: decoded representation of layer1 → LIF ensemble
            layer2 = nengo.Ensemble(
                n_neurons=self.n_neurons_layer2,
                dimensions=self.latent_dim,
                neuron_type=nengo.LIF(),
                seed=self.seed,
            )
            # Connect via decoded activity: layer1 → layer2 (latent_dim dims)
            nengo.Connection(
                layer1, layer2,
                synapse=self.synapse,
            )

            # Probe the decoded output of layer2
            self._probe = nengo.Probe(layer2, synapse=self.synapse)

        self._model = model

    def _run_sim(self, flat_image: np.ndarray) -> np.ndarray:
        """Run one presentation-time simulation and return the final probe value."""
        nengo = self._nengo

        if self._model is None:
            self._build_model(flat_image)

        # Update the current image fed to the network
        self._current_input[0] = flat_image

        with nengo.Simulator(self._model, seed=self.seed, progress_bar=False) as sim:
            sim.run(self.presentation_time)

        # Take the last decoded sample as the latent vector
        return sim.data[self._probe][-1].astype(float)


# ---------------------------------------------------------------------------
# SNNFeatureEncoder  — latent vector → unitary SP
# ---------------------------------------------------------------------------

class SNNFeatureEncoder:
    """Maps SNN latent representations to unitary Semantic Pointers.

    Works in two modes:

    **Mode 1 — external latent vectors** (framework-agnostic):
        Extract a latent vector from *any* SNN (snnTorch, Norse, BindsNET,
        Lava, …) and call ``encode(latent)`` to obtain a unitary SP.

    **Mode 2 — built-in Nengo SNN**:
        Pass a :class:`NengoSNN` as the ``nengo_snn`` argument.  Then call
        ``extract_and_encode(image)`` to go directly from an image to a
        unitary SP in one step.

    The ``encode()`` method is duck-type-compatible with
    :class:`~sspslam.perception.ImageFeatureEncoder`, so this encoder can be
    passed directly to
    :func:`~sspslam.networks.slam.get_slam_input_functions_from_features`
    without any changes to the SLAM network code.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the SNN latent vectors.
    ssp_dim : int
        Target SP dimensionality — must match the ``ssp_dim`` of the
        ``SSPSpace`` used in the SLAM network.
    seed : int or None
        Seed for the random projection matrix.  Fix to ensure reproducibility.
    normalize_input : bool
        If True (default) each latent vector is L2-normalised before
        projection.  Recommended unless the SNN already produces unit vectors.
    nengo_snn : NengoSNN or None
        Optional :class:`NengoSNN` instance.  Required only for
        ``extract_and_encode()``.
    """

    def __init__(self,
                 latent_dim: int,
                 ssp_dim: int,
                 seed=None,
                 normalize_input: bool = True,
                 nengo_snn=None):
        self.latent_dim = int(latent_dim)
        self.ssp_dim = int(ssp_dim)
        self.normalize_input = normalize_input
        self.nengo_snn = nengo_snn

        rng = np.random.RandomState(seed)

        # (ssp_dim × latent_dim) row-normalised random projection matrix
        W = rng.randn(self.ssp_dim, self.latent_dim)
        W = W / np.linalg.norm(W, axis=1, keepdims=True)
        self.projection = W   # (ssp_dim, latent_dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, latent_vectors: np.ndarray) -> np.ndarray:
        """Project SNN latent vector(s) to unitary SPs.

        Parameters
        ----------
        latent_vectors : np.ndarray
            Shape ``(latent_dim,)`` for a single vector, or
            ``(n, latent_dim)`` for a batch.

        Returns
        -------
        np.ndarray
            Shape ``(ssp_dim,)`` or ``(n, ssp_dim)``.  Each row is a
            real-valued, unitary SP (all Fourier-component magnitudes = 1).
        """
        v = np.atleast_2d(latent_vectors).astype(float)
        if v.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected latent_dim={self.latent_dim}, got {v.shape[1]}"
            )

        if self.normalize_input:
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            v = v / np.maximum(norms, 1e-8)

        sp = v @ self.projection.T      # (n, ssp_dim)
        sp = self._make_unitary(sp)
        return sp.squeeze()

    def encode_sequence(self, latent_sequence: np.ndarray) -> np.ndarray:
        """Encode a time-series of per-landmark latent vectors.

        Parameters
        ----------
        latent_sequence : np.ndarray
            Shape ``(T, N_landmarks, latent_dim)``.

        Returns
        -------
        np.ndarray
            Shape ``(T, N_landmarks, ssp_dim)``.
        """
        T, N, L = latent_sequence.shape
        if L != self.latent_dim:
            raise ValueError(
                f"Expected latent_dim={self.latent_dim}, got {L}"
            )
        flat = latent_sequence.reshape(T * N, L)
        encoded = self.encode(flat)   # (T*N, ssp_dim) or may squeeze if T*N==1
        encoded = np.atleast_2d(encoded)
        return encoded.reshape(T, N, self.ssp_dim)

    def extract_and_encode(self, image: np.ndarray) -> np.ndarray:
        """Image → NengoSNN latent → unitary SP.

        Requires that a :class:`NengoSNN` was passed as ``nengo_snn`` at
        construction time.

        Parameters
        ----------
        image : np.ndarray
            Shape (H, W) or (H, W, C).

        Returns
        -------
        np.ndarray
            Shape (ssp_dim,).
        """
        if self.nengo_snn is None:
            raise RuntimeError(
                "extract_and_encode() requires a NengoSNN.  "
                "Pass nengo_snn=NengoSNN(...) to SNNFeatureEncoder()."
            )
        latent = self.nengo_snn.extract(image)   # (latent_dim,)
        return self.encode(latent)               # (ssp_dim,)

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
