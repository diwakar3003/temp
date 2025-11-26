"""
Unscented Kalman Filter (UKF) for UWB Tracking with Occlusion Handling
========================================================================

This module implements a 2D UKF that continues to predict the position of a person
even when they are out of the camera frame (occlusion). The UKF maintains the
trajectory estimate during occlusion periods, which is crucial for re-identification
when the person re-enters the frame.

Key Features:
- Constant Velocity motion model for 2D tracking
- Continues prediction during occlusion (no camera updates)
- Tracks prediction uncertainty (covariance) during occlusion
- Provides smoothed position and velocity estimates
"""

import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from scipy.linalg import block_diag


class UWBUKFOcclusion:
    """
    2D Unscented Kalman Filter for UWB tracking with occlusion handling.
    
    State Vector: [x, v_x, y, v_y] (position and velocity in 2D)
    
    Motion Model: Constant Velocity
    - x(k+1) = x(k) + v_x(k) * dt
    - v_x(k+1) = v_x(k)
    - y(k+1) = y(k) + v_y(k) * dt
    - v_y(k+1) = v_y(k)
    
    Observation Model (when camera tracklet is available):
    - z = [x, y] (direct 2D position)
    
    During occlusion: Only prediction step is executed, no update step.
    """
    
    def __init__(self, dt=1/30.0, process_noise_std=0.5, measurement_noise_std=2.0):
        """
        Initialize the UKF for 2D UWB tracking.
        
        Args:
            dt: Time step (default: 1/30 seconds for 30 FPS)
            process_noise_std: Standard deviation of process noise
            measurement_noise_std: Standard deviation of measurement noise
        """
        self.dt = dt
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std
        
        # State dimension: [x, v_x, y, v_y]
        self.state_dim = 4
        
        # Measurement dimension: [x, y]
        self.measurement_dim = 2
        
        # Create sigma points
        self.sigma_points = MerweScaledSigmaPoints(
            n=self.state_dim,
            alpha=1e-3,
            beta=2.0,
            kappa=0.0
        )
        
        # Initialize UKF
        self.ukf = UKF(
            dim_x=self.state_dim,
            dim_z=self.measurement_dim,
            dt=dt,
            hx=self._h,  # Measurement function
            fx=self._f,  # State transition function
            points=self.sigma_points
        )
        
        # Set process noise covariance (Q)
        q = self.process_noise_std ** 2
        self.ukf.Q = np.eye(self.state_dim) * q
        
        # Set measurement noise covariance (R)
        r = self.measurement_noise_std ** 2
        self.ukf.R = np.eye(self.measurement_dim) * r
        
        # Initial state covariance (P)
        self.ukf.P = np.eye(self.state_dim) * 1.0
        
        # Initial state: [x, v_x, y, v_y]
        self.ukf.x = np.array([0.0, 0.0, 0.0, 0.0])
        
        # Track occlusion state
        self.is_occluded = False
        self.occlusion_start_frame = None
        self.occlusion_duration = 0
        
        # Store history for analysis
        self.state_history = []
        self.covariance_history = []
        self.measurement_history = []
        self.occlusion_history = []
    
    def _f(self, x, dt):
        """
        State transition function (Constant Velocity model).
        
        Args:
            x: Current state [x, v_x, y, v_y]
            dt: Time step
            
        Returns:
            Next state
        """
        # Constant velocity model
        F = np.array([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1]
        ])
        return F @ x
    
    def _h(self, x):
        """
        Measurement function (direct position observation).
        
        Args:
            x: Current state [x, v_x, y, v_y]
            
        Returns:
            Measurement [x, y]
        """
        # Observe only position, not velocity
        return np.array([x[0], x[2]])
    
    def predict(self, frame_number):
        """
        Predict step: Update state estimate without measurement.
        This is used both during normal operation and during occlusion.
        
        Args:
            frame_number: Current frame number (for tracking)
        """
        self.ukf.predict()
        
        # Store state and covariance
        self.state_history.append(self.ukf.x.copy())
        self.covariance_history.append(self.ukf.P.copy())
        self.occlusion_history.append(self.is_occluded)
    
    def update(self, measurement, frame_number):
        """
        Update step: Correct state estimate with measurement.
        This is only called when a camera tracklet is available (not occluded).
        
        Args:
            measurement: Measurement [x, y]
            frame_number: Current frame number (for tracking)
        """
        self.ukf.update(measurement)
        
        # Store measurement
        self.measurement_history.append(measurement.copy())
        
        # Exit occlusion if we were occluded
        if self.is_occluded:
            self.is_occluded = False
            print(f"[Frame {frame_number}] Re-identification successful! Occlusion ended after {self.occlusion_duration} frames.")
    
    def start_occlusion(self, frame_number):
        """
        Mark the start of an occlusion period.
        
        Args:
            frame_number: Frame number when occlusion started
        """
        if not self.is_occluded:
            self.is_occluded = True
            self.occlusion_start_frame = frame_number
            self.occlusion_duration = 0
            print(f"[Frame {frame_number}] Occlusion started. UKF will continue predicting...")
    
    def end_occlusion(self, frame_number):
        """
        Mark the end of an occlusion period (when person re-enters frame).
        
        Args:
            frame_number: Frame number when occlusion ended
        """
        if self.is_occluded:
            self.occlusion_duration = frame_number - self.occlusion_start_frame
            print(f"[Frame {frame_number}] Occlusion ended. Duration: {self.occlusion_duration} frames ({self.occlusion_duration / 30:.2f} seconds)")
    
    def get_state(self):
        """Get current state estimate [x, v_x, y, v_y]."""
        return self.ukf.x.copy()
    
    def get_position(self):
        """Get current position estimate [x, y]."""
        return np.array([self.ukf.x[0], self.ukf.x[2]])
    
    def get_velocity(self):
        """Get current velocity estimate [v_x, v_y]."""
        return np.array([self.ukf.x[1], self.ukf.x[3]])
    
    def get_covariance(self):
        """Get current covariance matrix (4x4)."""
        return self.ukf.P.copy()
    
    def get_position_covariance(self):
        """Get position covariance (2x2 submatrix)."""
        # Extract position covariance from full covariance matrix
        return self.ukf.P[np.ix_([0, 2], [0, 2])]
    
    def get_uncertainty(self):
        """
        Get uncertainty measure (trace of position covariance).
        Higher values indicate higher uncertainty.
        """
        pos_cov = self.get_position_covariance()
        return np.trace(pos_cov)
    
    def initialize_state(self, initial_position, initial_velocity=None):
        """
        Initialize the UKF state with known position and velocity.
        
        Args:
            initial_position: Initial position [x, y]
            initial_velocity: Initial velocity [v_x, v_y] (default: [0, 0])
        """
        if initial_velocity is None:
            initial_velocity = np.array([0.0, 0.0])
        
        self.ukf.x = np.array([
            initial_position[0],
            initial_velocity[0],
            initial_position[1],
            initial_velocity[1]
        ])
        
        # Reset covariance to initial uncertainty
        self.ukf.P = np.eye(self.state_dim) * 1.0


class OcclusionTracker:
    """
    Tracks occlusion events and manages multiple UKF instances for different persons.
    """
    
    def __init__(self, dt=1/30.0):
        """
        Initialize the occlusion tracker.
        
        Args:
            dt: Time step (default: 1/30 seconds for 30 FPS)
        """
        self.dt = dt
        self.ukf_filters = {}  # Dictionary to store UKF for each person
        self.active_tracklets = {}  # Current active tracklets (frame -> tracklet_id)
        self.missing_tracklets = {}  # Tracklets that disappeared (tracklet_id -> disappearance_frame)
    
    def create_filter(self, person_id, initial_position, initial_velocity=None):
        """
        Create a new UKF filter for a person.
        
        Args:
            person_id: Unique identifier for the person
            initial_position: Initial position [x, y]
            initial_velocity: Initial velocity [v_x, v_y]
        """
        ukf = UWBUKFOcclusion(dt=self.dt)
        ukf.initialize_state(initial_position, initial_velocity)
        self.ukf_filters[person_id] = ukf
    
    def predict_all(self, frame_number):
        """Predict for all active filters."""
        for person_id, ukf in self.ukf_filters.items():
            ukf.predict(frame_number)
    
    def update_filter(self, person_id, measurement, frame_number):
        """Update a specific filter with a measurement."""
        if person_id in self.ukf_filters:
            self.ukf_filters[person_id].update(measurement, frame_number)
    
    def mark_occlusion(self, person_id, frame_number):
        """Mark a person as occluded."""
        if person_id in self.ukf_filters:
            self.ukf_filters[person_id].start_occlusion(frame_number)
            self.missing_tracklets[person_id] = frame_number
    
    def mark_visible(self, person_id, frame_number):
        """Mark a person as visible again."""
        if person_id in self.ukf_filters:
            self.ukf_filters[person_id].end_occlusion(frame_number)
            if person_id in self.missing_tracklets:
                del self.missing_tracklets[person_id]
    
    def get_predicted_position(self, person_id):
        """Get predicted position for a person."""
        if person_id in self.ukf_filters:
            return self.ukf_filters[person_id].get_position()
        return None
    
    def get_filter_state(self, person_id):
        """Get full state for a person."""
        if person_id in self.ukf_filters:
            return self.ukf_filters[person_id].get_state()
        return None
    
    def get_position_uncertainty(self, person_id):
        """Get position uncertainty for a person."""
        if person_id in self.ukf_filters:
            return self.ukf_filters[person_id].get_uncertainty()
        return None
