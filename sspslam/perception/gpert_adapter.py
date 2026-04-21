import cv2
import numpy as np

class SimulatedGPERTAdapter:
    """
    Simulates the outputs of a GPERT 3DGS process by reading perfectly synchronized
    depth maps from a simulator and extracting metric depth for tracked event features.
    """
    def __init__(self, max_landmarks=30, fov_x=60.0):
        self.max_landmarks = max_landmarks
        self.fov_x = fov_x
        
        self.orb = cv2.ORB_create()
        self.feature_params = dict(maxCorners=100, qualityLevel=0.01, minDistance=10, blockSize=7)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        self.tracks = []
        self.next_track_id = 0
        self.prev_gray = None
        self.history = []

    def _get_tracks_arrays(self):
        if not self.tracks:
            return None
        return np.float32([tr['p'] for tr in self.tracks]).reshape(-1, 1, 2)

    def detect_and_track(self, gray_obs):
        if not self.tracks and self.prev_gray is None:
            corners = cv2.goodFeaturesToTrack(gray_obs, mask=None, **self.feature_params)
            if corners is not None:
                for p in corners:
                    self.tracks.append({'id': self.next_track_id, 'p': p[0]})
                    self.next_track_id += 1
            self.prev_gray = gray_obs
            return
            
        p0 = self._get_tracks_arrays()
        if p0 is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray_obs, p0, None, **self.lk_params)
            new_tracks = []
            for i, (tr, is_good) in enumerate(zip(self.tracks, st)):
                if is_good[0] == 1:
                    tr['p'] = p1[i][0]
                    new_tracks.append(tr)
            self.tracks = new_tracks
            
        if len(self.tracks) < self.max_landmarks:
            mask = np.ones_like(gray_obs)
            for tr in self.tracks:
                cv2.circle(mask, (int(tr['p'][0]), int(tr['p'][1])), 10, 0, -1)
            
            new_params = dict(self.feature_params)
            new_params['maxCorners'] = self.max_landmarks - len(self.tracks)
            corners = cv2.goodFeaturesToTrack(gray_obs, mask=mask, **new_params)
            if corners is not None:
                for p in corners:
                    self.tracks.append({'id': self.next_track_id, 'p': p[0]})
                    self.next_track_id += 1
                    
        self.prev_gray = gray_obs

    def process_frame(self, event_frame, simulated_depth_map):
        """
        Extracts metrics using the simulated GPERT depth map.
        simulated_depth_map: 2D array of floats representing metric distance from camera.
        """
        self.detect_and_track(event_frame)
        frame_data = {}
        
        if not self.tracks:
            self.history.append(frame_data)
            return frame_data
            
        cv_kps = [cv2.KeyPoint(x=float(tr['p'][0]), y=float(tr['p'][1]), size=15) for tr in self.tracks]
        cv_kps, descs = self.orb.compute(event_frame, cv_kps)
        
        height, width = event_frame.shape[:2]
        
        for i, tr in enumerate(self.tracks):
            if descs is None or i >= len(descs):
                break
                
            u, v = int(tr['p'][0]), int(tr['p'][1])
            if u < 0 or u >= width or v < 0 or v >= height:
                continue
                
            # Query the simulated GPERT depth!
            # Using bilinear interpolation or nearest neighbor
            try:
                # clamp coordinates
                u_c = max(0, min(u, width - 1))
                v_c = max(0, min(v, height - 1))
                depth = float(simulated_depth_map[v_c, u_c])
                
                # If depth is 0 (invalid/sky), use a fallback huge distance
                if depth <= 0.0:
                    depth = 50.0 
                    
            except IndexError:
                depth = 1.0 # Fallback
                
            nx = (u / float(width)) - 0.5
            bearing = nx * (self.fov_x * np.pi / 180.0) 
            
            # GPERT true continuous metric vector
            vec = np.array([depth * np.sin(bearing), depth * np.cos(bearing)])
            
            frame_data[tr['id']] = {
                'dist': depth,
                'bearing': bearing,
                'vector': vec,
                'desc': descs[i].astype(np.float32)
            }
            
        self.history.append(frame_data)
        return frame_data
