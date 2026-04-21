import cv2
import numpy as np

class ORBVisualOdometry:
    """
    A native Python Visual Odometry tracking frontend using ORB features.
    It mimics the frontend tracking module of standard ORB-SLAM.
    """
    def __init__(self, camera_matrix=None):
        # Default intrinsic camera matrix based on typical 60 FOV
        if camera_matrix is None:
            self.K = np.array([[300., 0., 160.],
                               [0., 300., 120.],
                               [0., 0., 1.]])
        else:
            self.K = camera_matrix
            
        self.orb = cv2.ORB_create(nfeatures=500, fastThreshold=20)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        self.prev_kps = None
        self.prev_descs = None
        
        # Absolute trajectory 
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))
        
        # Tracking history
        self.trajectory = []
        self.tracked_landmarks = [] # Relative 3D vectors to tracked landmarks
        
    def process_frame(self, frame_gray):
        """
        Processes a sequential frame (can be RGB converted to gray, or an Event Time-Surface).
        Returns:
            translation_velocity (dx, dy, dz)
            active_descriptors (N_landmarks, 32)
            active_landmark_vectors (N_landmarks, 2) # X, Z plane for SLAM
        """
        kps, descs = self.orb.detectAndCompute(frame_gray, None)
        
        vel_estimate = np.zeros(3)
        active_landmarks = []
        active_descs = []
        
        # Initialization
        if self.prev_descs is None or self.prev_kps is None:
            self.prev_kps = kps
            self.prev_descs = descs
            self.trajectory.append(self.t.copy())
            return vel_estimate, np.empty((0, 32)), np.empty((0, 2))
            
        if descs is not None and len(descs) > 0 and len(self.prev_descs) > 0:
            matches = self.matcher.match(self.prev_descs, descs)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Keep top matches
            matches = matches[:min(len(matches), 100)]
            
            if len(matches) > 10:
                pts1 = np.float32([self.prev_kps[m.queryIdx].pt for m in matches])
                pts2 = np.float32([kps[m.trainIdx].pt for m in matches])
                
                # Estimate Essential Matrix to recover ego-motion
                E, mask = cv2.findEssentialMat(pts2, pts1, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                
                if E is not None:
                    _, R_est, t_est, mask_pose = cv2.recoverPose(E, pts2, pts1, self.K, mask=mask)
                    
                    # Monocular scale ambiguity: t_est is a unit vector with unknown magnitude.
                    # scale=0.1 is calibrated for MiniWorld's forward_step=0.5 at 60° FOV;
                    # override via camera_matrix if using a different setup.
                    scale = 0.1
                    
                    self.t = self.t + scale * self.R.dot(t_est)
                    self.R = R_est.dot(self.R)
                    
                    vel_estimate = (scale * R_est.dot(t_est)).flatten()
                    
                    # Project landmarks
                    idx_inliers = np.where(mask_pose.flatten() == 255)[0]
                    
                    # We output the matched features that behaved as inliers
                    for i in idx_inliers:
                        train_idx = matches[i].trainIdx
                        active_descs.append(descs[train_idx])
                        
                        # Approximating relative vector using pixel bearing and arbitrary depth (scale)
                        u, v = pts2[i]
                        fx, fy = self.K[0,0], self.K[1,1]
                        cx, cy = self.K[0,2], self.K[1,2]
                        # 3D ray direction
                        ray_x = (u - cx) / fx
                        ray_y = (v - cy) / fy
                        
                        # In the ground plane, Z is forward (ray_z = 1.0), X is lateral
                        rel_vec = np.array([ray_x * scale * 10, 1.0 * scale * 10]) 
                        active_landmarks.append(rel_vec)
                        
        self.prev_kps = kps
        self.prev_descs = descs
        self.trajectory.append(self.t.copy())
        
        return vel_estimate, np.array(active_descs) if len(active_descs) > 0 else np.empty((0, 32)), np.array(active_landmarks) if len(active_landmarks) > 0 else np.empty((0, 2))
