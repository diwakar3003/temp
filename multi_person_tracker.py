"""
Multi-Person Tracker for Opt-in Camera System
=============================================

This module manages multiple UWB-tracked persons using individual UKF instances.
It handles the prediction and update steps for all active UWB tags.
"""

import numpy as np
from typing import Dict, Tuple, List
from uwb_ukf_2d_streaming import StreamingUWBUKF2D


class MultiPersonTracker:
    """
    Manages a collection of StreamingUWBUKF2D instances, one for each UWB tag.
    """
    
    def __init__(self, dt: float, buffer_size: int):
        """
        Args:
            dt: Time step (1/FPS).
            buffer_size: Size of the UKF state buffer.
        """
        self.dt = dt
        self.buffer_size = buffer_size
        self.trackers: Dict[int, StreamingUWBUKF2D] = {}
        
    def initialize_trackers(self, initial_states: Dict[int, Tuple[np.ndarray, np.ndarray]]):
        """
        Initializes a UKF for each UWB ID.
        
        Args:
            initial_states: Dict of {uwb_id: (initial_pos, initial_vel)}
        """
        for uwb_id, (initial_pos, initial_vel) in initial_states.items():
            ukf = StreamingUWBUKF2D(dt=self.dt, buffer_size=self.buffer_size)
            ukf.initialize_state(initial_pos, initial_vel)
            self.trackers[uwb_id] = ukf
            
    def predict_all(self, frame: int, timestamp: float):
        """
        Runs the prediction step for all active UKF trackers.
        """
        for uwb_id, ukf in self.trackers.items():
            ukf.predict(frame, timestamp)
            
    def update_all_uwb(self, uwb_measurements: List[Dict]):
        """
        Runs the update step for UKF trackers based on UWB measurements.
        
        Args:
            uwb_measurements: List of UWB measurements for the current frame.
        """
        for measurement in uwb_measurements:
            uwb_id = measurement['id']
            uwb_pos = measurement['position']
            frame = measurement['frame']
            timestamp = measurement['timestamp']
            
            if uwb_id in self.trackers:
                self.trackers[uwb_id].update_camera(uwb_pos, frame, timestamp)
            # Note: In a real system, we would also handle new UWB tags appearing here.
            
    def get_ukf_states_for_matching(self) -> Dict[int, Dict]:
        """
        Returns the latest UKF state (position and covariance) for all active trackers.
        
        Returns:
            Dict of {uwb_id: {'position': np.ndarray, 'covariance': np.ndarray}}
        """
        states = {}
        for uwb_id, ukf in self.trackers.items():
            states[uwb_id] = {
                'position': ukf.get_position(),
                'covariance': ukf.get_position_covariance()
            }
        return states
    
    def get_ukf_metrics(self) -> Dict[int, Dict]:
        """
        Returns the latest UKF metrics for logging.
        """
        metrics = {}
        for uwb_id, ukf in self.trackers.items():
            metrics[uwb_id] = {
                'uncertainty': ukf.get_uncertainty(),
                'timestamp': ukf.get_latest_metrics().timestamp
            }
        return metrics
    
    def get_max_processing_time(self) -> float:
        """
        Returns the maximum processing time across all UKF instances.
        """
        max_time = 0.0
        for ukf in self.trackers.values():
            max_time = max(max_time, ukf.get_max_processing_time())
        return max_time

def create_multi_person_tracker(dt: float, fps: int) -> MultiPersonTracker:
    """Helper function to create a MultiPersonTracker instance."""
    # Buffer size is set to 5 seconds of frames
    buffer_size = fps * 5
    return MultiPersonTracker(dt=dt, buffer_size=buffer_size)
