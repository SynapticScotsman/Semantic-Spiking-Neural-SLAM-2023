import cv2
import numpy as np
import math

class EventOdometry:
    """
    Event-based Visual Odometry (EVO)
    Computes camera ego-motion purely from matching 2D event features across
    accumulated event time-surfaces using Epipolar geometry (Essential Matrix).
    """
    
    def __init__(self, focal_length=715.0, pp=(320.0, 240.0)):
        # Camera intrinsic parameters (approximate defaults for 640x480 resolution)
        self.focal_length = focal_length
        self.pp = pp
        
        self.feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=15, blockSize=7)
        self.lk_params = dict(winSize=(21, 21), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        
        self.prev_gray = None
        self.p0 = None
        
        # Tracked pose
        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))
        self.velocities = []

    def get_absolute_scale(self):
        # Without stereo or IMU, monocular VO translation scale is ambiguous.
        # Returning 1.0 for normalized relative direction, or tuned constant.
        return 1.0 

    def process_frame(self, current_frame):
        """
        Calculates the relative velocity and ego-motion given an event frame.
        current_frame: 2D numpy array (H, W) uint8
        """
        # Initialization
        if self.prev_gray is None:
            self.p0 = cv2.goodFeaturesToTrack(current_frame, mask=None, **self.feature_params)
            self.prev_gray = current_frame
            return np.array([0., 0., 0.]), np.array([0., 0., 0.])
            
        if self.p0 is None or len(self.p0) < 8:
            # Need to re-detect if we lost features
            self.p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
            if self.p0 is None or len(self.p0) < 8:
                self.prev_gray = current_frame
                return np.array([0., 0., 0.]), np.array([0., 0., 0.])

        # Optical Flow tracking
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, current_frame, self.p0, None, **self.lk_params)
        
        # Select good points
        good_new = p1[st == 1]
        good_old = self.p0[st == 1]

        if len(good_new) < 8:
            # Not enough points for Essential Matrix
            self.prev_gray = current_frame
            self.p0 = cv2.goodFeaturesToTrack(current_frame, mask=None, **self.feature_params)
            return np.array([0., 0., 0.]), np.array([0., 0., 0.])

        # Compute Essential Matrix
        E, mask = cv2.findEssentialMat(
            good_new, good_old, 
            self.focal_length, self.pp, 
            cv2.RANSAC, 0.999, 1.0, None)
            
        if E is None or E.shape != (3, 3):
            # Fallback if EM fails
            self.prev_gray = current_frame
            self.p0 = cv2.goodFeaturesToTrack(current_frame, mask=None, **self.feature_params)
            return np.array([0., 0., 0.]), np.array([0., 0., 0.])

        # Recover pose (Rotation and Translation relative to previous frame)
        _, R_rel, t_rel, mask = cv2.recoverPose(E, good_new, good_old, focal=self.focal_length, pp=self.pp)

        absolute_scale = self.get_absolute_scale()
        
        # Since t_rel is a unit vector up to sign, recoverPose fixes the sign using cheirality check
        rel_vel = t_rel.flatten() * absolute_scale
        
        # Accumulate absolute pose
        if absolute_scale > 0.1:
            self.cur_t = self.cur_t + absolute_scale * self.cur_R.dot(t_rel)
            self.cur_R = self.cur_R.dot(R_rel)
            
        # Convert R_rel to euler angles for rotational velocity
        # R_rel is close to identity for small movements
        sy = math.sqrt(R_rel[0,0] * R_rel[0,0] +  R_rel[1,0] * R_rel[1,0])
        singular = sy < 1e-6
        if not singular:
            x_rot = math.atan2(R_rel[2,1] , R_rel[2,2])
            y_rot = math.atan2(-R_rel[2,0], sy)
            z_rot = math.atan2(R_rel[1,0], R_rel[0,0])
        else:
            x_rot = math.atan2(-R_rel[1,2], R_rel[1,1])
            y_rot = math.atan2(-R_rel[2,0], sy)
            z_rot = 0
            
        rot_vel = np.array([x_rot, y_rot, z_rot])
            
        # Renew features
        self.prev_gray = current_frame
        self.p0 = good_new.reshape(-1, 1, 2)
        
        # If tracking count drops below threshold, detect more
        if self.p0.shape[0] < 80:
            new_features = cv2.goodFeaturesToTrack(current_frame, mask=None, **self.feature_params)
            if new_features is not None:
                self.p0 = np.vstack((self.p0, new_features))
                
        return rel_vel, rot_vel

