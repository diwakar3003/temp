"""
Real-Time Trajectory Matching with Latency Optimization
======================================================

This module adapts the 2D trajectory matching logic for real-time streaming
inference. It focuses on minimizing latency by using a sliding window for
similarity calculation and the Hungarian algorithm for optimal assignment.

Key Features:
- Sliding window for time-averaged similarity calculation
- Latency-optimized Mahalanobis distance calculation
- Real-time assignment using the Hungarian algorithm
"""

import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.optimize import linear_sum_assignment
from collections import deque
import time
from typing import List, Dict, Tuple


class RealTimeTrajectoryMatcher:
    """
    Performs real-time trajectory matching between UKF-smoothed UWB positions
    and camera tracklets using a sliding window approach.
    """
    
    def __init__(self, 
                 dt: float = 1/30.0,
                 matching_rate_fps: int = 10,
                 similarity_window_frames: int = 30,
                 cost_threshold: float = 0.5):
        """
        Initialize the real-time trajectory matcher.
        
        Args:
            dt: Time step (seconds)
            matching_rate_fps: Frequency at which to run the LAP solver (e.g., 10 FPS)
            similarity_window_frames: Number of frames to use for time-averaged similarity
            cost_threshold: Maximum Mahalanobis distance (cost) for a valid match
        """
        self.dt = dt
        self.matching_rate_frames = int(30 / matching_rate_fps)  # Frames between LAP runs
        self.similarity_window_frames = similarity_window_frames
        self.cost_threshold = cost_threshold
        self.last_lap_frame = -self.matching_rate_frames
        
        # Buffer to store recent UKF states and camera tracklet data
        self.ukf_state_buffer: Dict[int, deque] = {} # uwb_id -> deque of (frame, position, covariance)
        self.camera_tracklet_buffer = {}  # tracklet_id -> deque of (frame, position, covariance)
        
        # Current assignments
        self.current_assignments = {}  # UWB_ID -> Tracklet_ID
        self.assignment_history = deque(maxlen=100)
    
    def update_ukf_state(self, frame: int, uwb_id: int, position: np.ndarray, covariance: np.ndarray):
        """
        Update the buffer with the latest UKF state.
        
        Args:
            frame: Current frame number
            uwb_id: ID of the UWB tag
            position: UKF position estimate [x, y]
            covariance: UKF position covariance (2x2)
        """
        if uwb_id not in self.ukf_state_buffer:
            self.ukf_state_buffer[uwb_id] = deque(maxlen=self.similarity_window_frames)
            
        self.ukf_state_buffer[uwb_id].append({
            'frame': frame,
            'position': position.copy(),
            'covariance': covariance.copy()
        })
    
    def update_camera_tracklets(self, frame: int, tracklets: List[Dict]):
        """
        Update the buffer with the latest camera tracklet data.
        
        Args:
            frame: Current frame number
            tracklets: List of tracklet data [{'id': int, 'position': np.ndarray}]
        """
        # Clear old data from tracklet buffer
        frames_to_keep = set(range(frame - self.similarity_window_frames + 1, frame + 1))
        
        # Update existing tracklets and add new ones
        active_tracklet_ids = set()
        for tracklet in tracklets:
            tracklet_id = tracklet['id']
            position = tracklet['position']
            
            if tracklet_id not in self.camera_tracklet_buffer:
                self.camera_tracklet_buffer[tracklet_id] = deque(maxlen=self.similarity_window_frames)
            
            self.camera_tracklet_buffer[tracklet_id].append({
                'frame': frame,
                'position': position.copy()
            })
            active_tracklet_ids.add(tracklet_id)
        
        # Remove tracklets that have disappeared from the buffer
        tracklets_to_remove = []
        for tracklet_id, buffer in self.camera_tracklet_buffer.items():
            # Check if the tracklet is still active in the current frame
            if tracklet_id not in active_tracklet_ids:
                # If not active, check if its last frame is outside the window
                if len(buffer) > 0 and buffer[-1]['frame'] < frames_to_keep.pop() - 1:
                    tracklets_to_remove.append(tracklet_id)
        
        for tracklet_id in tracklets_to_remove:
            del self.camera_tracklet_buffer[tracklet_id]
    
    def _calculate_mahalanobis_cost(self, ukf_data: Dict, tracklet_data: Dict) -> float:
        """
        Calculate the Mahalanobis distance between UKF and camera position.
        
        Args:
            ukf_data: {'frame', 'position', 'covariance'}
            tracklet_data: {'frame', 'position'}
            
        Returns:
            Mahalanobis distance (cost)
        """
        # Ensure frames are the same (should be guaranteed by synchronization)
        if ukf_data['frame'] != tracklet_data['frame']:
            return np.inf
        
        ukf_pos = ukf_data['position']
        cam_pos = tracklet_data['position']
        cov = ukf_data['covariance']
        
        # Calculate Mahalanobis distance
        try:
            # Add a small epsilon to the diagonal for numerical stability
            cov_stable = cov + np.eye(2) * 1e-6
            inv_cov = np.linalg.inv(cov_stable)
            
            diff = ukf_pos - cam_pos
            mahal_dist = np.sqrt(diff @ inv_cov @ diff.T)
            return mahal_dist
        except np.linalg.LinAlgError:
            return np.inf
    
    def _calculate_time_averaged_similarity(self, uwb_id: int, tracklet_id: int) -> float:
        """
        Calculate the time-averaged similarity (inverse of cost) over the sliding window.
        
        Args:
            ukf_id: Placeholder for UWB ID (assuming one UWB tag for simplicity)
            tracklet_id: ID of the camera tracklet
            
        Returns:
            Time-averaged similarity score (higher is better)
        """
        if uwb_id not in self.ukf_state_buffer or tracklet_id not in self.camera_tracklet_buffer:
            return -np.inf
        
        ukf_buffer = self.ukf_state_buffer[uwb_id]
        tracklet_buffer = self.camera_tracklet_buffer[tracklet_id]
        
        total_cost = 0.0
        count = 0
        
        # Create a set of frames present in the tracklet buffer for quick lookup
        tracklet_frames = {item['frame']: item for item in tracklet_buffer}
        
        # Iterate over the UKF state buffer (which represents the UWB trajectory)
        for ukf_data in ukf_buffer:
            ukf_frame = ukf_data['frame']
            
            # Find the corresponding camera tracklet data for this frame
            cam_data = tracklet_frames.get(ukf_frame)
            
            if cam_data is not None:
                cost = self._calculate_mahalanobis_cost(ukf_data, cam_data)
                
                # Only include costs below the threshold in the average
                if cost < self.cost_threshold:
                    total_cost += cost
                    count += 1
        
        if count == 0:
            return -np.inf  # No valid overlap or all costs above threshold
        
        # Time-averaged cost (Mahalanobis distance)
        avg_cost = total_cost / count
        
        # Convert cost to similarity (lower cost -> higher similarity)
        # Similarity = 1 / (1 + cost)
        similarity = 1.0 / (1.0 + avg_cost)
        
        return similarity
    
    def run_lap_assignment(self, frame: int) -> Dict[int, int]:
        """
        Runs the Linear Assignment Problem (LAP) solver at the matching rate.
        
        Args:
            frame: Current frame number
            
        Returns:
            Dictionary of current assignments {UWB_ID: Tracklet_ID}
        """
        if frame - self.last_lap_frame < self.matching_rate_frames:
            return self.current_assignments
        
        start_time = time.time()
        
        # Get all UWB IDs that have recent UKF state data
        uwb_ids = list(self.ukf_state_buffer.keys())
        tracklet_ids = list(self.camera_tracklet_buffer.keys())
        
        if not tracklet_ids:
            self.current_assignments = {}
            self.last_lap_frame = frame
            return self.current_assignments
        
        # Build cost matrix (negative similarity, as LAP minimizes cost)
        n_uwb = len(uwb_ids)
        n_cam = len(tracklet_ids)
        
        # Initialize cost matrix with a high cost (low similarity)
        MAX_COST = 1000.0 # A large finite number to represent infeasible assignments
        cost_matrix = np.full((n_uwb, n_cam), MAX_COST)
        
        for i, uwb_id in enumerate(uwb_ids):
            for j, tracklet_id in enumerate(tracklet_ids):
                similarity = self._calculate_time_averaged_similarity(uwb_id, tracklet_id)
                
                # Cost is negative similarity (minimize cost = maximize similarity)
                if similarity > -np.inf:
                    cost_matrix[i, j] = -similarity
                else:
                    cost_matrix[i, j] = MAX_COST # Replace -inf similarity with MAX_COST
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Update current assignments
        new_assignments = {}
        for i, j in zip(row_ind, col_ind):
            uwb_id = uwb_ids[i]
            tracklet_id = tracklet_ids[j]
            
            # Check if the assignment is a real match or a padded MAX_COST
            if cost_matrix[i, j] < MAX_COST:
                similarity = -cost_matrix[i, j]
                
                # Only assign if similarity is above a threshold (e.g., 1 / (1 + cost_threshold))
                min_similarity = 1.0 / (1.0 + self.cost_threshold)
                
                if similarity >= min_similarity:
                    new_assignments[uwb_id] = tracklet_id
                    self.assignment_history.append({
                    'frame': frame,
                    'uwb_id': uwb_id,
                    'tracklet_id': tracklet_id,
                    'similarity': similarity
                })
        
        self.current_assignments = new_assignments
        self.last_lap_frame = frame
        
        lap_latency = (time.time() - start_time) * 1000
        # print(f"LAP run at frame {frame}. Latency: {lap_latency:.2f}ms")
        
        return self.current_assignments
    
    def get_current_assignments(self) -> Dict[int, int]:
        """Get the latest assignments."""
        return self.current_assignments
    
    def get_assignment_history(self) -> deque:
        """Get the history of assignments."""
        return self.assignment_history
    
    def reset(self):
        """Reset all buffers and state."""
        for buffer in self.ukf_state_buffer.values():
            buffer.clear()
        self.ukf_state_buffer.clear()
        self.camera_tracklet_buffer.clear()
        self.current_assignments = {}
        self.assignment_history.clear()
        self.last_lap_frame = -self.matching_rate_frames
