"""
semantic_encoder.py
===================
Binds a visual appearance Semantic Pointer (SP) — produced by
``ImageFeatureEncoder`` — with a weighted semantic-class SP — drawn from
an ``SPSpace`` vocabulary — via circular convolution.

The resulting *compound* SP serves as the landmark identity in SSP-SLAM.
Because binding is bilinear, the compound SP has a well-defined relationship
to class-label queries:

    dot(bind(app, p1*CHAIR + p2*TABLE), bind(app, CHAIR))
        = dot(p1*CHAIR + p2*TABLE, CHAIR)
        ≈ p1

(Uses the fact that the appearance SP ``app`` is unitary, so binding with it
is an isometry on the space of SPs.)

This means:
  * **Query by class**: form ``bind(app_i, CHAIR_sp)`` for each landmark *i*
    and use the normalised sum as a key in the associative memory.  The
    memory output will have high spatial similarity near landmarks whose
    CHAIR probability is high.
  * **Query by appearance**: form ``bind(app_novel, sem_expected)`` where
    ``sem_expected`` is the weighted-sum semantic SP for the expected class
    distribution.  The memory output peaks near the landmark whose visual
    appearance best matches ``app_novel``.

Usage
-----
    from sspslam import SPSpace
    from sspslam.perception import ImageFeatureEncoder, SemanticLandmarkEncoder

    sp_space = SPSpace(domain_size=4, dim=97, seed=0)
    img_enc  = ImageFeatureEncoder(feat_dim=128, ssp_dim=97, seed=0)

    enc = SemanticLandmarkEncoder(
        image_encoder=img_enc,
        sp_space=sp_space,
        label_names=["CHAIR", "TABLE", "DOOR", "WINDOW"],
    )

    landmark_sp = enc.encode(
        img_features=some_128d_array,
        label_probabilities={"CHAIR": 0.8, "TABLE": 0.2},
    )

    # Query helpers
    chair_sp   = enc.label_sp("CHAIR")          # single-class SP
    app_sp     = enc.get_appearance_sp(feat)    # appearance SP only
    sem_sp     = enc.get_semantic_sp({"CHAIR": 0.6, "TABLE": 0.4})  # weighted semantic
"""

import numpy as np


class SemanticLandmarkEncoder:
    """Produces bound appearance × semantic SPs for SSP-SLAM landmarks.

    Parameters
    ----------
    image_encoder : ImageFeatureEncoder
        Converts raw feature vectors into unitary appearance SPs.
    sp_space : SPSpace
        Semantic vocabulary. Each row of ``sp_space.vectors`` corresponds to
        a label in ``label_names``.  Must have
        ``sp_space.domain_size >= len(label_names)``.
    label_names : list of str
        Human-readable class labels (e.g. ``["CHAIR", "TABLE", "DOOR"]``).
        Their order determines which row of ``sp_space.vectors`` they map to.

    Attributes
    ----------
    vocab : dict
        Maps each label name to its index in ``sp_space.vectors``.
    """

    def __init__(self, image_encoder, sp_space, label_names):
        label_names = list(label_names)
        if len(label_names) > sp_space.domain_size:
            raise ValueError(
                f"SPSpace has domain_size={sp_space.domain_size}, but "
                f"{len(label_names)} label_names were provided."
            )
        self.image_encoder = image_encoder
        self.sp_space = sp_space
        self.label_names = label_names
        self.vocab = {name: i for i, name in enumerate(label_names)}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, img_features: np.ndarray,
               label_probabilities: dict) -> np.ndarray:
        """Encode a landmark observation as a bound appearance × semantic SP.

        The returned SP is:
            bind(appearance_sp, semantic_sp)

        where ``semantic_sp = sum_j prob_j * label_j_sp``.

        Because ``appearance_sp`` is unitary, querying with
        ``bind(appearance_sp, label_j_sp)`` gives a dot-product response
        proportional to ``prob_j`` (see module docstring).

        Parameters
        ----------
        img_features : np.ndarray
            Raw feature vector of shape ``(feat_dim,)``.
        label_probabilities : dict
            Mapping of label name → weight, e.g. ``{"CHAIR": 0.8, "TABLE": 0.2}``.
            Weights need not sum to 1, but the interpretation as probabilities
            is most natural when they do.

        Returns
        -------
        np.ndarray
            Bound SP of shape ``(sp_space.dim,)``.
        """
        appearance_sp = self.get_appearance_sp(img_features)
        semantic_sp = self.get_semantic_sp(label_probabilities)
        result = self.sp_space.bind(
            np.atleast_2d(appearance_sp),
            np.atleast_2d(semantic_sp),
        )
        return result.squeeze()

    def get_semantic_sp(self, label_probabilities: dict) -> np.ndarray:
        """Compute the weighted sum of label SPs.

        Parameters
        ----------
        label_probabilities : dict
            Mapping of label name → weight.

        Returns
        -------
        np.ndarray
            Shape ``(sp_space.dim,)``.

        Raises
        ------
        KeyError
            If any label in ``label_probabilities`` is not in ``vocab``.
        """
        sp = np.zeros(self.sp_space.dim)
        for label, prob in label_probabilities.items():
            if label not in self.vocab:
                raise KeyError(
                    f"Unknown label: '{label}'. "
                    f"Known labels: {self.label_names}"
                )
            sp += float(prob) * self.sp_space.vectors[self.vocab[label]]
        return sp

    def get_appearance_sp(self, img_features: np.ndarray) -> np.ndarray:
        """Encode a raw feature vector to a unitary appearance SP.

        Parameters
        ----------
        img_features : np.ndarray
            Shape ``(feat_dim,)``.

        Returns
        -------
        np.ndarray
            Shape ``(sp_space.dim,)``.
        """
        return self.image_encoder.encode(img_features)

    def label_sp(self, label_name: str) -> np.ndarray:
        """Return the SP vector for a single semantic class label.

        Parameters
        ----------
        label_name : str
            Must be one of the names passed to ``__init__``.

        Returns
        -------
        np.ndarray
            Shape ``(sp_space.dim,)``.
        """
        if label_name not in self.vocab:
            raise KeyError(
                f"Unknown label: '{label_name}'. "
                f"Known labels: {self.label_names}"
            )
        return self.sp_space.vectors[self.vocab[label_name]]

    def class_query_sp(self, label_name: str,
                       appearance_sps: np.ndarray) -> np.ndarray:
        """Build a normalised class-query SP for use with the associative memory.

        Forms ``normalize(sum_i bind(app_i, label_sp))`` over all landmark
        appearances.  When used as a key in the associative memory (which stores
        ``bind(app_i, sem_i) → location_i``), the output spatial similarity at
        each point scales with the corresponding landmark's probability for
        ``label_name``.

        Parameters
        ----------
        label_name : str
            The semantic class to query for.
        appearance_sps : np.ndarray
            Shape ``(n_landmarks, sp_space.dim)``.  Precomputed appearance SPs
            for all landmarks.

        Returns
        -------
        np.ndarray
            Shape ``(sp_space.dim,)``, the normalised class-query SP.
        """
        cls_sp = self.label_sp(label_name)
        cls_2d = np.atleast_2d(cls_sp)
        # bind each app_i with the class SP, then sum
        bound = self.sp_space.bind(appearance_sps, np.tile(cls_2d, (len(appearance_sps), 1)))
        total = bound.sum(axis=0)
        norm = np.linalg.norm(total)
        return total / max(norm, 1e-12)

    def appearance_query_sp(self, img_features: np.ndarray,
                            semantic_sp: np.ndarray) -> np.ndarray:
        """Build a normalised appearance-query SP for use with the associative memory.

        Forms ``normalize(bind(app_novel, semantic_sp))``.  When the semantic
        SP is chosen to match the expected class distribution of the target
        landmark, querying the associative memory with this vector retrieves the
        location of the landmark whose visual appearance most closely matches
        ``img_features``.

        Parameters
        ----------
        img_features : np.ndarray
            Shape ``(feat_dim,)``.  The novel visual observation.
        semantic_sp : np.ndarray
            Shape ``(sp_space.dim,)``.  Expected semantic SP for the target
            class (e.g. from ``get_semantic_sp``).

        Returns
        -------
        np.ndarray
            Shape ``(sp_space.dim,)``.
        """
        app_sp = self.get_appearance_sp(img_features)
        bound = self.sp_space.bind(
            np.atleast_2d(app_sp),
            np.atleast_2d(semantic_sp),
        ).squeeze()
        norm = np.linalg.norm(bound)
        return bound / max(norm, 1e-12)
