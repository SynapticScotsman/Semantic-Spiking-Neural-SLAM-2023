"""Object-centric VSA cognitive map: a scene map plus per-object view files.

Two memories, not one blob:

**Scene map** -- which instances exist and where they are::

    M_scene = (1/N) sum_i  ID_i (*) S_allo(p_i)

``S_allo`` is an ordinary (aperiodic) SSP of world position: the same
navigation index as SSP-SLAM.  ``ID_i`` is a random unitary atom, minted when
an instance is first seen.

**Object file** -- what one instance looks like from which side::

    V_i = (1/K) sum_k  c(z_k) (*) S_view(phi_k)

``S_view`` is a *periodic* FPE of object-centred azimuth, so the view code
wraps after a full orbit and binding rotates you around the object.

Keeping them apart is not tidiness.  Folding the view book into the spatial
term (``ID (*) S_allo (*) V``) means the residue of a "where is it" query is
``S_allo (*) V`` rather than ``S_allo``, and binding by a random-looking
vector destroys the spatial peak -- the position becomes undecodable.  The
same blob also breaks the view query in the other direction, since unbinding
``ID (*) S_view`` then leaves ``S_allo (*) c`` rather than ``c``.  A single
combined vector is still available via :meth:`ObjectCentricMap.object_vector`,
which superposes rather than binds the two roles; ``mode='bind'`` reproduces
the fully-bound form so the difference can be measured (see
``experiments/run_object_map.py``).

Capacity scales as *objects x distinct views*, not as pixels: both memories
are bundles of unit vectors, so the read-out SNR falls off roughly as
``1/sqrt(number of superposed terms)``.
"""

import numpy as np

from ..sspspace import HexagonalSSPSpace
from .appearance import AppearanceCodec
from .geometry import (camera_bearing, view_azimuth,
                       view_azimuth_elevation, wrap_angle)
from .objectfile import ObjectFile
from .viewspace import CircularSSPSpace
from .vsa import (AtomVocab, VectorCodebook, bind, bundle, cosine, normalize,
                  unbind)

__all__ = ["ObjectCentricMap", "PlaceQuery", "ViewQuery",
           "ViewLocalisation"]


class PlaceQuery:
    """Result of a "where is it" query."""

    def __init__(self, position, score, grid=None, heatmap=None):
        self.position = position
        self.score = score
        self.grid = grid
        self.heatmap = heatmap

    def __repr__(self):
        pos = np.round(self.position, 3) if self.position is not None else None
        return f"PlaceQuery(position={pos}, score={self.score:.3f})"


class ViewQuery:
    """Result of a "what does it look like from there" query."""

    def __init__(self, residue, matches, phi, bearing=None):
        self.residue = residue
        self.matches = matches
        self.phi = phi
        self.bearing = bearing

    @property
    def best(self):
        return self.matches[0] if self.matches else None

    def __repr__(self):
        return (f"ViewQuery(phi={np.round(self.phi, 3)}, "
                f"best={self.best})")


class ViewLocalisation:
    """Result of asking which way round an object you are seeing it.

    Attributes
    ----------
    phi : np.ndarray
        Peak of the likelihood -- the estimated viewing direction.
    score : float
        Height of that peak.
    margin : float
        Peak minus the best competing peak outside the main lobe.  A small
        margin means the object looks the same from more than one side, which
        is a property of the object, not a failure of the estimate.
    angles, field : np.ndarray
        The full likelihood over the view manifold.
    """

    def __init__(self, phi, score, margin, angles, field):
        self.phi = phi
        self.score = score
        self.margin = margin
        self.angles = angles
        self.field = field

    def __repr__(self):
        return (f"ViewLocalisation(phi={np.round(np.rad2deg(self.phi), 1)} deg, "
                f"score={self.score:.3f}, margin={self.margin:+.3f})")


class ObjectCentricMap:
    """A scene map of object files, written from a walk and read by unbinding.

    Parameters
    ----------
    feat_dim : int
        Dimension of the incoming crop embeddings.
    ssp_dim : int
        Requested vector dimension.  The allocentric space may round this to
        the nearest value its tiling supports; the rounded value is used
        everywhere and is available as :attr:`dim`.
    domain_dim : int
        2 for a floor plan, 3 for a volume.
    bounds : array_like or None
        ``(domain_dim, 2)`` world bounds, used to build the decoding grid.
        Defaults to ``[-10, 10]`` per axis.
    view_dims : int
        1 for an azimuth circle around each object, 2 for a view sphere
        (azimuth and elevation); 2 requires ``domain_dim == 3``.
    max_harmonic : int
        Angular resolution of the view circle (see :mod:`.viewspace`).
        Also, and less obviously, the **capacity** setting: it fixes how many
        distinct frequencies the code has, and that is what decides how many
        views a single bundle can hold.  The default of 8 is kept for
        reproducibility but is on the wrong side of the measured optimum --
        FINDINGS.md sec.16 E2 finds ``4`` better at every K tested, and it is
        the difference between the object file losing to a list of stored
        views and tying one.  Prefer 4 for new work unless you have measured
        otherwise on your own front end.
    length_scale : float
        Length scale of the allocentric SSP, in world units.
    grid_step : float
        Spacing of the position-decoding grid, in world units.
    assoc_radius : float
        Detections further than this from a known object's position can never
        be associated with it.
    assoc_threshold : float
        Appearance-key cosine above which a detection joins an existing
        object.  Below it, a detection inside ``same_place_radius`` still
        associates (an object seen from a new side looks nothing like itself).
    same_place_radius : float
        Tight positional gate used when appearance does not match.
    merge_tol : float
        Angular distance below which a new view merges into a stored one.
    stats_warmup : int
        Number of crops after which the appearance statistics freeze.  They
        have to settle *before* many keys are minted, because a key encoded
        under one centring cannot be compared with one encoded under another;
        call :meth:`AppearanceCodec.fit` up front when a sample is available
        and this warm-up never fires.
    seed : int or None

    Examples
    --------
    >>> import numpy as np
    >>> m = ObjectCentricMap(feat_dim=16, ssp_dim=151, seed=0)
    >>> z = np.random.default_rng(0).standard_normal(16)
    >>> oid, is_new = m.observe(robot_pos=[0, 0], obj_pos=[2, 1], embedding=z)
    >>> is_new
    True
    """

    def __init__(self, feat_dim, ssp_dim=151, domain_dim=2, bounds=None,
                 view_dims=1, max_harmonic=8, length_scale=1.0,
                 grid_step=0.25, assoc_radius=1.0, assoc_threshold=0.55,
                 same_place_radius=0.35, merge_tol=np.deg2rad(20.0),
                 stats_warmup=200, seed=None):
        if view_dims == 2 and domain_dim < 3:
            raise ValueError("a view sphere (view_dims=2) needs domain_dim=3")

        self.rng = np.random.default_rng(seed)
        self.domain_dim = int(domain_dim)
        self.view_dims = int(view_dims)

        self.allo_space = HexagonalSSPSpace(
            domain_dim=self.domain_dim, ssp_dim=int(ssp_dim),
            length_scale=length_scale, rng=self.rng,
            domain_bounds=self._default_bounds(bounds),
        )
        self.dim = int(self.allo_space.ssp_dim)
        self.view_space = CircularSSPSpace(
            domain_dim=self.view_dims, ssp_dim=self.dim,
            max_harmonic=max_harmonic, rng=self.rng,
        )
        self.appearance = AppearanceCodec(feat_dim, self.dim, seed=seed)

        self.ids = AtomVocab(self.dim, seed=seed)
        self.classes = AtomVocab(self.dim, seed=None if seed is None
                                 else seed + 1)
        self.view_codebook = VectorCodebook(self.dim)

        self.objects = {}
        self.bounds = self.allo_space.domain_bounds
        self.grid_step = float(grid_step)
        self.assoc_radius = float(assoc_radius)
        self.assoc_threshold = float(assoc_threshold)
        self.same_place_radius = float(same_place_radius)
        self.merge_tol = merge_tol
        self.stats_warmup = int(stats_warmup)

        self._grid_pts = None
        self._grid_ssps = None
        self._scene_cache = None
        self._class_cache = None
        self._next_id = 0

    def _default_bounds(self, bounds):
        if bounds is None:
            return np.tile([[-10.0, 10.0]], (self.domain_dim, 1))
        bounds = np.atleast_2d(np.asarray(bounds, dtype=float))
        if bounds.shape != (self.domain_dim, 2):
            raise ValueError(
                f"bounds must have shape {(self.domain_dim, 2)}, "
                f"got {bounds.shape}")
        return bounds

    # ------------------------------------------------------------------
    # Writing -- fill the map from a walk
    # ------------------------------------------------------------------

    def observe(self, robot_pos, obj_pos, embedding, robot_yaw=0.0,
                class_name=None, obj_id=None, obj_yaw=None, timestamp=None,
                update_position=True):
        """Fold one detection into the map.

        The caller supplies what a detector and a SLAM front end already
        produce: where the robot is, where the detection is in the world, and
        a crop embedding.  The viewpoint angle is derived from the two poses
        rather than stored as ego data.

        Parameters
        ----------
        robot_pos : array_like
            World position of the camera.
        obj_pos : array_like
            World position of the detection.
        embedding : array_like
            Crop embedding of ``feat_dim`` elements.
        robot_yaw : float
            Robot heading; only used to report the camera bearing back.
        class_name : str or None
            Detector class, if you have one.  Gates data association and
            populates the class map.
        obj_id : str or None
            Force association with a known instance, bypassing the matcher.
        obj_yaw : float or None
            Heading of the object's front.  Set once per instance; ``None``
            keeps whatever the instance already has (0 for a new one).
        timestamp : float or None
        update_position : bool
            Fold ``obj_pos`` into the object's running position estimate.

        Returns
        -------
        (str, bool)
            The instance id, and whether it was newly minted.
        """
        robot_pos = np.asarray(robot_pos, dtype=float).reshape(-1)
        obj_pos = np.asarray(obj_pos, dtype=float).reshape(-1)
        key = self.appearance.encode(embedding)

        is_new = False
        if obj_id is None:
            obj_id = self._associate(obj_pos, key, class_name)
        if obj_id is None or obj_id not in self.objects:
            obj_id = self._mint(obj_pos, class_name, obj_yaw, obj_id)
            is_new = True

        obj = self.objects[obj_id]
        if obj_yaw is not None:
            obj.yaw = float(obj_yaw)
        if class_name is not None and obj.class_name is None:
            obj.class_name = class_name
            self.classes.mint(class_name)
            self._class_cache = None

        if update_position and not is_new:
            n = max(obj.n_observations, 1)
            obj.position = (obj.position * n + obj_pos) / (n + 1)
            self._scene_cache = None
            self._class_cache = None

        phi = self.view_angle(obj.position, robot_pos, obj.yaw)
        idx = obj.add_view(phi, key, merge_tol=self.merge_tol)
        self.view_codebook.add((obj_id, idx), obj.views[idx].key)
        if self.appearance.n_seen < self.stats_warmup:
            self.appearance.partial_fit(np.asarray(embedding, dtype=float))
        if timestamp is not None and not obj.history:
            obj.history.append((timestamp, np.array(obj.position)))
        return obj_id, is_new

    def view_angle(self, obj_pos, robot_pos, obj_yaw=0.0):
        """Object-centred viewpoint angle, matching this map's view space."""
        if self.view_dims == 2:
            return np.array(view_azimuth_elevation(obj_pos, robot_pos, obj_yaw))
        return np.array([view_azimuth(obj_pos, robot_pos, obj_yaw)])

    def move_object(self, obj_id, new_position, timestamp=None):
        """Relocate an instance, keeping its view book.

        This is what the object file buys over a snapshot map: the chair
        still looks the same from each side after somebody pushes it across
        the room, so only ``S_allo`` needs rewriting.
        """
        self.objects[obj_id].set_position(new_position, timestamp)
        self._scene_cache = None
        self._class_cache = None

    def _mint(self, obj_pos, class_name, obj_yaw, obj_id=None):
        if obj_id is None:
            obj_id = f"{class_name or 'obj'}_{self._next_id:03d}"
            self._next_id += 1
        id_vec = self.ids.mint(obj_id)
        if class_name is not None:
            self.classes.mint(class_name)
        self.objects[obj_id] = ObjectFile(
            obj_id=obj_id, id_vec=id_vec,
            position=np.asarray(obj_pos, dtype=float).reshape(-1),
            yaw=0.0 if obj_yaw is None else float(obj_yaw),
            class_name=class_name,
        )
        self._scene_cache = None
        self._class_cache = None
        return obj_id

    def _associate(self, obj_pos, key, class_name):
        """Match a detection to a known instance, or return None to mint one."""
        best_id, best_sim, nearest_id, nearest_d = None, -np.inf, None, np.inf
        for oid, obj in self.objects.items():
            if class_name is not None and obj.class_name is not None \
                    and obj.class_name != class_name:
                continue
            d = float(np.linalg.norm(obj.position - obj_pos))
            if d > self.assoc_radius:
                continue
            if d < nearest_d:
                nearest_id, nearest_d = oid, d
            if obj.n_views:
                sim = float(np.max(cosine(key, obj.keys)))
                if sim > best_sim:
                    best_id, best_sim = oid, sim
        if best_id is not None and best_sim >= self.assoc_threshold:
            return best_id
        # A known object seen from an unseen side looks like nothing on file,
        # so fall back to a tight positional gate before minting a duplicate.
        if nearest_id is not None and nearest_d <= self.same_place_radius:
            return nearest_id
        return None

    # ------------------------------------------------------------------
    # The memories
    # ------------------------------------------------------------------

    def scene_vector(self):
        """``(1/N) sum_i ID_i (*) S_allo(p_i)`` -- the navigation index."""
        if self._scene_cache is None:
            if not self.objects:
                self._scene_cache = np.zeros(self.dim)
            else:
                terms = np.stack([
                    bind(o.id_vec, self.allo_space.encode(o.position))
                    for o in self.objects.values()])
                self._scene_cache = bundle(terms)
        return self._scene_cache

    def class_vector(self):
        """``(1/N) sum_i CLASS_i (*) S_allo(p_i)`` -- "where are the chairs"."""
        if self._class_cache is None:
            terms = [bind(self.classes[o.class_name],
                          self.allo_space.encode(o.position))
                     for o in self.objects.values() if o.class_name is not None]
            self._class_cache = (bundle(np.stack(terms)) if terms
                                 else np.zeros(self.dim))
        return self._class_cache

    def object_vector(self, obj_id, mode="bundle"):
        """One vector holding both roles for a single object.

        Parameters
        ----------
        mode : {'bundle', 'bind'}
            ``'bundle'`` gives ``ID (*) unit(S_allo + V)``: unbinding ``ID``
            leaves ``S_allo`` plus view-book crosstalk, so the position still
            decodes, and unbinding ``ID (*) S_view(phi)`` still leaves the
            appearance key plus crosstalk.  ``'bind'`` gives the fully bound
            ``ID (*) S_allo (*) V``, which is included only so the failure can
            be measured -- neither read-out survives it.
        """
        obj = self.objects[obj_id]
        allo = self.allo_space.encode(obj.position).reshape(-1)
        view = obj.view_book(self.view_space)
        if mode == "bundle":
            return bind(obj.id_vec, normalize(allo + view))
        if mode == "bind":
            return bind(bind(obj.id_vec, allo), view)
        raise ValueError("mode must be 'bundle' or 'bind'")

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def where_is(self, obj_id, memory=None, return_heatmap=False):
        """*Where is this instance?*  Unbind ``ID``, correlate over space."""
        memory = self.scene_vector() if memory is None else memory
        residue = unbind(memory, self.ids[obj_id])
        return self._decode_place(residue, return_heatmap)

    def where_are(self, class_name, return_heatmap=False):
        """*Where are all the chairs?*  Unbind ``CLASS`` from the class map."""
        residue = unbind(self.class_vector(), self.classes[class_name])
        return self._decode_place(residue, return_heatmap)

    def what_is_at(self, position, top_k=3, memory=None):
        """*What is at this place?*  Unbind ``S_allo``, clean up the residue."""
        memory = self.scene_vector() if memory is None else memory
        allo = self.allo_space.encode(np.asarray(position, dtype=float))
        residue = unbind(memory, allo.reshape(-1))
        return self.ids.clean_up(residue, top_k=top_k)

    def view_of(self, obj_id, phi, top_k=3, restrict_to_object=False,
                memory=None):
        """*What does this side look like?*  Unbind ``S_view(phi)``.

        Parameters
        ----------
        phi : float or array_like
            Object-centred viewpoint angle (azimuth, or azimuth+elevation).
        restrict_to_object : bool
            Clean up against only this object's stored views.  Leave it off
            to check that the residue beats every other object's views too,
            which is the honest test.
        memory : np.ndarray or None
            Read from an arbitrary vector instead of the object's own file --
            e.g. the output of :meth:`object_vector` or :meth:`orbit`.
        """
        phi = np.atleast_1d(np.asarray(phi, dtype=float))
        book = (self.objects[obj_id].view_book(self.view_space)
                if memory is None else memory)
        residue = unbind(book, self.view_space.encode(phi))
        restrict = ((lambda lab: lab[0] == obj_id) if restrict_to_object
                    else None)
        matches = self.view_codebook.clean_up(residue, top_k=top_k,
                                              restrict=restrict)
        return ViewQuery(residue=residue, matches=matches, phi=phi)

    def expected_view_from(self, obj_id, robot_pos, robot_yaw=0.0, top_k=3,
                           restrict_to_object=False):
        """*What should I see of it from here?*

        Computes the viewpoint angle from the two poses, then asks the object
        file.  Also reports the camera bearing -- where in the field of view
        the object sits -- which is a different angle and is never stored.
        """
        obj = self.objects[obj_id]
        phi = self.view_angle(obj.position, robot_pos, obj.yaw)
        out = self.view_of(obj_id, phi, top_k=top_k,
                           restrict_to_object=restrict_to_object)
        out.bearing = camera_bearing(obj.position, robot_pos, robot_yaw)
        return out

    def identify(self, embedding, top_k=3, key=None, weight_by_count=False):
        """*What am I looking at?*  Score the crop against every prototype.

        Deliberately **not** routed through the view books.  Naming an object
        and reading its viewpoint are opposite problems: the first wants the
        part of the appearance that does not change as you walk round, the
        second wants the part that does.  Asking one bundle to do both is
        what FINDINGS.md sec.16 E0 measured failing -- identification through
        the view book drops as K grows, because the bound terms interfere.

        sec.16 E1: 0.89 here against 0.44 through the view book at
        ``ssp_dim=151``, on ten objects with twelve views each.  The usual
        pipeline is therefore ``identify`` to pick the instance, then
        :meth:`localise_view` on that instance to read the side.

        Returns a list of ``(obj_id, cosine)``, best first.
        """
        if key is None:
            key = self.appearance.encode(embedding)
        key = np.asarray(key, dtype=float).reshape(-1)
        names = [o for o in self.objects if self.objects[o].n_views]
        if not names:
            return []
        protos = np.stack([self.objects[n].prototype(weight_by_count)
                           for n in names])
        sims = cosine(key, protos)
        order = np.argsort(sims)[::-1][:top_k]
        return [(names[i], float(sims[i])) for i in order]

    def recognise(self, embedding, n_per_dim=720, key=None):
        """Identify, then localise on the winner.  The two-stage read-out.

        Returns ``(obj_id, id_score, ViewLocalisation)``, or ``None`` if the
        map holds no object with any views yet.
        """
        if key is None:
            key = self.appearance.encode(embedding)
        ranked = self.identify(None, top_k=1, key=key)
        if not ranked:
            return None
        obj_id, score = ranked[0]
        return obj_id, score, self.localise_view(obj_id, None, n_per_dim,
                                                 key=key)

    def localise_view(self, obj_id, embedding, n_per_dim=720, key=None):
        """*I can see this thing -- which side of it am I looking at?*

        The view-circle twin of localising in space.  Localising in a room
        means unbinding ``ID`` from the scene map and correlating the residue
        against a grid of ``S_allo(x)``; the peak is where you are.
        Localising on an object means correlating an observed crop key
        against the object file read at every viewpoint; the peak is the
        direction you are looking from.  Same operation, different manifold,
        and no pose input -- appearance alone fixes the angle.

        The whole field comes back, not just the peak, because its shape is
        the informative part: a symmetric object gives several equal peaks
        and no amount of confidence weighting will separate them.  Treat it
        as a likelihood over viewpoint and fuse it with odometry rather than
        trusting the argmax.

        Parameters
        ----------
        obj_id : str
        embedding : array_like or None
            Crop embedding of the thing you are looking at.  Pass ``key``
            instead if you have already encoded it.
        n_per_dim : int
            Resolution of the returned field.

        Returns
        -------
        ViewLocalisation
            ``phi`` (peak), ``score``, ``margin`` over the runner-up peak,
            ``angles`` and ``field`` for the full likelihood.
        """
        if key is None:
            key = self.appearance.encode(embedding)
        book = self.objects[obj_id].view_book(self.view_space)
        angles, field = self.view_space.view_likelihood(
            book, np.asarray(key).reshape(-1), n_per_dim=n_per_dim)
        flat = field.reshape(-1)
        best = int(np.argmax(flat))
        peak = angles.reshape(-1, self.view_dims)[best] if self.view_dims > 1 \
            else np.array([angles[best]])

        # Runner-up = best score outside the main lobe, which is what tells
        # you whether the object is symmetric enough to alias.
        lobe = self.view_space.lobe_width()
        far = np.max(np.abs(wrap_angle(
            angles.reshape(-1, self.view_dims) - peak)), axis=1) > 2 * lobe
        runner = float(np.max(flat[far])) if far.any() else -np.inf
        return ViewLocalisation(phi=peak, score=float(flat[best]),
                                margin=float(flat[best] - runner),
                                angles=angles, field=field)

    def orbit(self, view_code, delta):
        """Rotate a view code around the object by ``delta``, with no image.

        ``S_view`` is periodic, so this is exact modulo the bundling noise
        already in ``view_code``: orbiting by a full turn is the identity.
        """
        delta = np.atleast_1d(np.asarray(delta, dtype=float))
        return bind(view_code, self.view_space.encode(delta))

    def verify(self, obj_id, robot_pos, embedding, robot_yaw=0.0):
        """Cosine between the predicted appearance from here and what is seen.

        The loop-closure test an object map makes cheap: predict, then
        compare against the live crop key.
        """
        obs = self.appearance.encode(embedding)
        pred = self.expected_view_from(obj_id, robot_pos, robot_yaw)
        return float(cosine(obs, pred.residue.reshape(1, -1))[0])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _decode_place(self, residue, return_heatmap=False):
        pts, ssps = self.place_grid()
        sims = ssps @ np.asarray(residue, dtype=float).reshape(-1)
        best = int(np.argmax(sims))
        return PlaceQuery(
            position=pts[best], score=float(sims[best]),
            grid=pts if return_heatmap else None,
            heatmap=sims if return_heatmap else None,
        )

    def place_grid(self):
        """Cached ``(points, ssps)`` grid used for decoding positions."""
        if self._grid_pts is None:
            axes = [np.arange(lo, hi + 1e-9, self.grid_step)
                    for lo, hi in self.bounds]
            mesh = np.meshgrid(*axes, indexing="ij")
            self._grid_pts = np.stack([m.ravel() for m in mesh], axis=-1)
            self._grid_ssps = self.allo_space.encode(self._grid_pts)
        return self._grid_pts, self._grid_ssps

    def summary(self):
        """Per-object counts, useful for checking that ``K`` stayed small."""
        return [
            {
                "id": o.obj_id,
                "class": o.class_name,
                "position": np.round(o.position, 3),
                "n_views": o.n_views,
                "n_observations": o.n_observations,
                "coverage": round(o.coverage(self.view_space), 3),
            }
            for o in self.objects.values()
        ]

    def __repr__(self):
        n_views = sum(o.n_views for o in self.objects.values())
        return (f"ObjectCentricMap(dim={self.dim}, "
                f"objects={len(self.objects)}, views={n_views})")
