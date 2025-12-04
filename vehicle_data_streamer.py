"""
Vehicle Data Streamer (Simulated)
=================================

This module simulates real-time data streams from a vehicle-mounted system,
providing asynchronous UWB and camera tracklet data for multiple persons.

Key Features:
- Simulates multiple persons with independent trajectories.
- Generates 30 FPS camera tracklets.
- Generates 3 FPS UWB measurements.
- Includes occlusions and distractor tracklets.
"""

import numpy as np
from typing import Generator, Dict, List, Tuple
import time


class VehicleDataStreamer:
    """
    Simulates real-time data streams for multiple persons.
    """
    
    def __init__(self, 
                 total_duration_sec: float = 20.0,
                 fps: int = 30,
                 uwb_rate_fps: int = 3):
        """
        Initialize the streamer.
        
        Args:
            total_duration_sec: Total duration of the simulation
            fps: Camera frame rate
            uwb_rate_fps: UWB measurement rate
        """
        self.fps = fps
        self.dt = 1.0 / fps
        self.total_frames = int(total_duration_sec * fps)
        self.uwb_interval = int(fps / uwb_rate_fps)
        self.current_frame = 0
        
        # Pre-calculate the ground truth trajectory for multiple persons
        self.gt_trajectories = self._generate_ground_truths()
        self.uwb_ids = [101, 102] # UWB Tags
        self.tracklet_ids = [1, 2, 3] # Camera Tracklet IDs
    
    def _generate_ground_truths(self) -> Dict[int, List[Dict]]:
        """
        Generates ground truth trajectories for multiple persons.
        
        UWB ID 101: Walks in a straight line, then turns. Occluded 150-250.
        UWB ID 102: Enters later (Frame 100), walks opposite direction.
        Distractor (ID 3): No UWB tag, only camera tracklet. Visible 50-150.
        """
        trajectories = {}
        
        # --- Person 1 (UWB ID 101) ---
        trajectory_101 = []
        x, y = 10.0, 50.0
        vx, vy = 0.5, 0.0
        for frame in range(self.total_frames):
            x += vx * self.dt
            y += vy * self.dt
            if frame > 300 and frame < 400:
                vy += 0.005 * self.dt
            elif frame >= 400 and frame < 500:
                vy -= 0.005 * self.dt
            
            # Simple noise for smooth movement
            x += np.random.normal(0, 0.01)
            y += np.random.normal(0, 0.01)
            
            trajectory_101.append({'frame': frame, 'timestamp': frame * self.dt, 'position': np.array([x, y]), 'velocity': np.array([vx, vy])})
        trajectories[101] = trajectory_101
        
        # --- Person 2 (UWB ID 102) ---
        trajectory_102 = []
        x, y = 90.0, 60.0
        vx, vy = -0.4, 0.0
        for frame in range(self.total_frames):
            if frame < 100:
                x_curr, y_curr = 90.0, 60.0 # Not yet in frame
            else:
                x += vx * self.dt
                y += vy * self.dt
                x_curr, y_curr = x, y
            
            trajectory_102.append({'frame': frame, 'timestamp': frame * self.dt, 'position': np.array([x_curr, y_curr]), 'velocity': np.array([vx, vy])})
        trajectories[102] = trajectory_102
        
        # --- Person 3 (Distractor - Tracklet ID 3) ---
        trajectory_3 = []
        x, y = 50.0, 30.0
        vx, vy = 0.0, 0.3
        for frame in range(self.total_frames):
            if 50 <= frame <= 150:
                x += vx * self.dt
                y += vy * self.dt
                x_curr, y_curr = x, y
            else:
                x_curr, y_curr = 50.0, 30.0
            
            trajectory_3.append({'frame': frame, 'timestamp': frame * self.dt, 'position': np.array([x_curr, y_curr]), 'velocity': np.array([vx, vy])})
        trajectories[3] = trajectory_3
        
        return trajectories
    
    def stream_data(self) -> Generator[Dict, None, None]:
        """
        Generator that yields data packets for each frame.
        
        Yields:
            A dictionary containing:
            - 'frame': current frame number
            - 'timestamp': current timestamp
            - 'camera_tracklets': list of simulated camera tracklets
            - 'uwb_measurements': list of simulated UWB measurements
        """
        
        for frame in range(self.total_frames):
            
            camera_tracklets = []
            uwb_measurements = []
            
            # --- Simulate UWB Measurements (3 FPS) ---
            if frame % self.uwb_interval == 0:
                for uwb_id in self.uwb_ids:
                    gt_data = self.gt_trajectories[uwb_id][frame]
                    
                    # Only generate UWB if person is active (e.g., after frame 100 for ID 102)
                    if uwb_id == 102 and frame < 100:
                        continue
                    
                    uwb_pos = gt_data['position'] + np.random.normal(0, 0.5, 2)  # UWB noise (higher)
                    uwb_measurements.append({
                        'id': uwb_id,
                        'position': uwb_pos,
                        'frame': frame,
                        'timestamp': gt_data['timestamp']
                    })
            
            # --- Simulate Camera Tracklets (30 FPS) ---
            
            # Tracklet 1 (Corresponds to UWB ID 101)
            gt_data_101 = self.gt_trajectories[101][frame]
            if not (150 <= frame <= 250): # Occlusion period for Person 1
                cam_pos = gt_data_101['position'] + np.random.normal(0, 0.1, 2)
                cam_vel = gt_data_101['velocity'] + np.random.normal(0, 0.05, 2)
                camera_tracklets.append({
                    'id': self.tracklet_ids[0], # Tracklet ID 1
                    'position': cam_pos,
                    'velocity': cam_vel
                })
            
            # Tracklet 2 (Corresponds to UWB ID 102)
            gt_data_102 = self.gt_trajectories[102][frame]
            if frame >= 100 and not (300 <= frame <= 400): # Enters at 100, Occlusion 300-400
                cam_pos = gt_data_102['position'] + np.random.normal(0, 0.1, 2)
                cam_vel = gt_data_102['velocity'] + np.random.normal(0, 0.05, 2)
                camera_tracklets.append({
                    'id': self.tracklet_ids[1], # Tracklet ID 2
                    'position': cam_pos,
                    'velocity': cam_vel
                })
            
            # Tracklet 3 (Distractor - No UWB)
            gt_data_3 = self.gt_trajectories[3][frame]
            if 50 <= frame <= 150:
                cam_pos = gt_data_3['position'] + np.random.normal(0, 0.1, 2)
                cam_vel = gt_data_3['velocity'] + np.random.normal(0, 0.05, 2)
                camera_tracklets.append({
                    'id': self.tracklet_ids[2], # Tracklet ID 3
                    'position': cam_pos,
                    'velocity': cam_vel
                })
            
            yield {
                'frame': frame,
                'timestamp': gt_data_101['timestamp'], # Use Person 1's timestamp as reference
                'camera_tracklets': camera_tracklets,
                'uwb_measurements': uwb_measurements
            }
            
            self.current_frame = frame
    
    def get_total_frames(self) -> int:
        """Get total number of frames."""
        return self.total_frames
    
    def get_fps(self) -> int:
        """Get camera frame rate."""
        return self.fps
    
    def get_uwb_rate(self) -> int:
        """Get UWB measurement rate."""
        return int(self.fps / self.uwb_interval)
    
    def get_dt(self) -> float:
        """Get time step."""
        return self.dt
    
    def get_initial_states(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Get initial position and velocity for all UWB IDs."""
        initial_states = {}
        for uwb_id in self.uwb_ids:
            initial_data = self.gt_trajectories[uwb_id][0]
            initial_states[uwb_id] = (initial_data['position'], initial_data['velocity'])
        return initial_states
    
    def get_uwb_ids(self) -> List[int]:
        """Get list of UWB IDs."""
        return self.uwb_ids
    
    def get_tracklet_ids(self) -> List[int]:
        """Get list of tracklet IDs."""
        return self.tracklet_ids

def create_streamer(duration: float = 20.0) -> VehicleDataStreamer:
    """Helper function to create a streamer instance."""
    return VehicleDataStreamer(total_duration_sec=duration)
