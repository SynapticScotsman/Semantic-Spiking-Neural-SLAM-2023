import cv2
import numpy as np

class MiniWorldVisionAdapter:
    """
    Bridges visual observations from MiniWorld (RGB + Depth) to the SSP-SLAM pipeline.
    It detects structural keypoints using Shi-Tomasi corners, tracks them across frames
    using Lucas-Kanade optical flow, and extracts ORB appearance descriptors.
    Spatial positions are estimated via the depth map.
    """
    
    def __init__(self, env, max_landmarks=30, view_radius=3.0):
        self.env = env
        self.max_landmarks = max_landmarks
        self.view_radius = view_radius
        
        # SLAM Network expects a fixed number of landmarks across the episode
        # We will track up to `max_landmarks` persistent features.
        
        # Detector and Extractor
        self.orb = cv2.ORB_create()
        self.feature_params = dict(maxCorners=100, qualityLevel=0.01, minDistance=10, blockSize=7)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        self.tracks = []           # List of active tracks: structure: {id: int, p: (u, v)}
        self.next_track_id = 0
        self.prev_gray = None
        
        # Stores history for SLAM array formatting later
        # Format: list of dictionaries mapping track_id -> dict(dist, bearing, vector, descriptor)
        self.history = []
        
        # Camera intrinsics (approximations from MiniWorld)
        self.fov_y = env.unwrapped.agent.cam_fov_y
        
    def _get_tracks_arrays(self):
        """Helper to get tracked points as (N, 1, 2) array for optical flow"""
        if not self.tracks:
            return None
        return np.float32([tr['p'] for tr in self.tracks]).reshape(-1, 1, 2)
        
    def detect_and_track_keypoints(self, gray_obs):
        """
        Track existing keypoints using optical flow and detect new ones if needed.
        """
        # If no tracks, detect fresh ones
        if not self.tracks and self.prev_gray is None:
            corners = cv2.goodFeaturesToTrack(gray_obs, mask=None, **self.feature_params)
            if corners is not None:
                for p in corners:
                    self.tracks.append({'id': self.next_track_id, 'p': p[0]})
                    self.next_track_id += 1
            self.prev_gray = gray_obs
            return
            
        # Track existing with Optical Flow
        p0 = self._get_tracks_arrays()
        if p0 is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray_obs, p0, None, **self.lk_params)
            
            new_tracks = []
            for i, (tr, is_good) in enumerate(zip(self.tracks, st)):
                if is_good[0] == 1:
                    tr['p'] = p1[i][0]
                    new_tracks.append(tr)
            self.tracks = new_tracks
            
        # If we lost too many, detect new ones
        if len(self.tracks) < self.max_landmarks:
            mask = np.ones_like(gray_obs)
            for tr in self.tracks:
                cv2.circle(mask, (int(tr['p'][0]), int(tr['p'][1])), 10, 0, -1) # Mask out existing
            
            new_params = dict(self.feature_params)
            new_params['maxCorners'] = self.max_landmarks - len(self.tracks)
            
            corners = cv2.goodFeaturesToTrack(gray_obs, mask=mask, **new_params)
            if corners is not None:
                for p in corners:
                    self.tracks.append({'id': self.next_track_id, 'p': p[0]})
                    self.next_track_id += 1
                    
        self.prev_gray = gray_obs

    def process_frame(self, rgb_obs, depth_obs):
        gray = cv2.cvtColor(rgb_obs, cv2.COLOR_RGB2GRAY)
        self.detect_and_track_keypoints(gray)
        
        frame_data = {}
        
        # Compute keypoints for ORB extraction
        if not self.tracks:
            self.history.append(frame_data)
            return frame_data
            
        cv_kps = [cv2.KeyPoint(x=float(tr['p'][0]), y=float(tr['p'][1]), size=15) for tr in self.tracks]
        
        # Extract ORB descriptors for the current keypoints
        cv_kps, descs = self.orb.compute(gray, cv_kps)
        
        height, width = depth_obs.shape[:2]
        
        for i, tr in enumerate(self.tracks):
            # Safe indexing into descriptor array
            if descs is None or i >= len(descs):
                break
                
            u, v = int(tr['p'][0]), int(tr['p'][1])
            if u < 0 or u >= width or v < 0 or v >= height:
                continue
                
            # Depth from miniworld (in meters)
            dist = depth_obs[v, u]
            if np.isscalar(dist):
                d = dist
            else:
                d = dist[0]
                
            if d > self.view_radius or d < 0.1:
                continue # Too far or too close
                
            # Compute bearing angle
            # v FOV is fov_y, u FOV is derived from aspect ratio
            aspect = width / height
            fov_x = self.fov_y * aspect
            
            # Map u from [0, width] to [-0.5, 0.5] then multiply by FOV
            nx = (u / float(width)) - 0.5
            bearing = nx * (fov_x * np.pi / 180.0) # approx horizontal bearing in radians
            
            # Compute 2D vector relative to agent heading (z is forward, x is right)
            # In SSP-SLAM, usually vector is 2D mapping (e.g. x, y)
            vec = np.array([d * np.sin(bearing), d * np.cos(bearing)])
            
            frame_data[tr['id']] = {
                'dist': d,
                'bearing': bearing,
                'vector': vec,
                'desc': descs[i].astype(np.float32)
            }
            
        self.history.append(frame_data)
        return frame_data

    def collect_episode(self, max_steps=100, policy=None):
        """
        Runs the environment for max_steps, processes frames, and returns SLAM arrays.
        Policy should be a function: action = policy(obs)
        """
        obs, info = self.env.reset()
        
        velocities = []
        path = [] # True path for error plotting
        
        last_pos = np.array([self.env.unwrapped.agent.pos[0], self.env.unwrapped.agent.pos[2]])
        path.append(last_pos)
        
        for t in range(max_steps):
            if policy:
                action = policy(obs)
            else:
                # Default policy: move forward with occasional turn
                action = 2 if t % 10 != 0 else 0
                
            obs, reward, term, trunc, info = self.env.step(action)
            
            # Agent velocity is difference in position 
            # Note: real agent only has proprioception (actions), but we'll cheat just a bit 
            # on velocity for the sake of path integration, or we could use odometry from actions.
            # SLAM Network needs dx/dt.
            current_pos = np.array([self.env.unwrapped.agent.pos[0], self.env.unwrapped.agent.pos[2]])
            dt_step = 0.05 # Assumed timestep
            vel = (current_pos - last_pos) / dt_step
            velocities.append(vel)
            
            last_pos = current_pos
            path.append(current_pos)
            
            # Get purely visual data
            depth = self.env.unwrapped.render_depth()
            self.process_frame(obs, depth)
            
            if term or trunc:
                break
                
        # Now format into the (T, N_landmarks, ...) arrays
        T = len(self.history)
        # Find which track ids were most stable (seen in most frames)
        track_counts = {}
        for frame in self.history:
            for tid in frame.keys():
                track_counts[tid] = track_counts.get(tid, 0) + 1
                
        # Select top `max_landmarks` most frequently seen keypoints
        sorted_tracks = sorted(track_counts.items(), key=lambda item: item[1], reverse=True)
        top_tracks = [t[0] for t in sorted_tracks[:self.max_landmarks]]
        actual_N = len(top_tracks)
            
        feat_dim = 32 # ORB descriptor is 32 bytes (but we converted to float32). 
                      
        vel_arr = np.array(velocities)
        vec_arr = np.zeros((T, self.max_landmarks, 2))
        feat_arr = np.zeros((T, self.max_landmarks, 32)) 
        
        # Fill arrays
        # If unseen, dist is > view_radius, so distance > view_radius acts as "unseen" flag.
        # We will make unseen vectors length = view_radius * 2
        for t, frame in enumerate(self.history):
            for n_i in range(self.max_landmarks):
                if n_i < actual_N:
                    tid = top_tracks[n_i]
                    if tid in frame:
                        vec_arr[t, n_i] = frame[tid]['vector']
                        feat_arr[t, n_i] = frame[tid]['desc']
                    else:
                        vec_arr[t, n_i] = np.array([self.view_radius * 2, self.view_radius * 2])
                else:
                    vec_arr[t, n_i] = np.array([self.view_radius * 2, self.view_radius * 2])
                    
        return vel_arr, vec_arr, feat_arr, np.array(path)
