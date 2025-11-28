"""
Vehicle Data Streamer (Simulated)
=================================

This module simulates real-time data streams from a vehicle-mounted system,
providing asynchronous UWB and camera tracklet data.

Key Features:
- Simulates a person walking alongside a moving vehicle.
- Generates 30 FPS camera tracklets.
- Generates 3 FPS UWB measurements.
- Uses a generator function to simulate continuous streaming.
"""

import numpy as np
from typing import Generator, Dict, List, Tuple
import time
from typing import Generator, Dict, List


class VehicleDataStreamer:
    """
    Simulates real-time data streams for a person walking alongside a vehicle.
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
        
        # Pre-calculate the ground truth trajectory
        self.gt_trajectory = self._generate_ground_truth()
    
    def _generate_ground_truth(self) -> List[Dict]:
        """
        Generates a ground truth trajectory for a person.
        
        Scenario: Person walks in a straight line, then turns slightly.
        """
        trajectory = []
        
        # Initial state
        x, y = 10.0, 50.0
        vx, vy = 0.5, 0.0
        
        for frame in range(self.total_frames):
            # Simple motion model
            x += vx * self.dt
            y += vy * self.dt
            
            # Introduce a slight turn after 10 seconds (300 frames)
            if frame > 300 and frame < 400:
                vy += 0.005 * self.dt  # Accelerate slightly in y direction
            elif frame >= 400 and frame < 500:
                vy -= 0.005 * self.dt  # Decelerate in y direction
            
            # Add noise to simulate real-world movement
            x += np.random.normal(0, 0.01)
            y += np.random.normal(0, 0.01)
            
            # Update velocity for next step
            if frame > 0:
                prev_x, prev_y = trajectory[-1]['position']
                vx = (x - prev_x) / self.dt
                vy = (y - prev_y) / self.dt
            
            trajectory.append({
                'frame': frame,
                'timestamp': frame * self.dt,
                'position': np.array([x, y]),
                'velocity': np.array([vx, vy])
            })
            
        return trajectory
    
    def stream_data(self) -> Generator[Dict, None, None]:
        """
        Generator that yields data packets for each frame.
        
        Yields:
            A dictionary containing:
            - 'frame': current frame number
            - 'timestamp': current timestamp
            - 'camera_tracklets': list of simulated camera tracklets
            - 'uwb_measurement': simulated UWB measurement or None
        """
        
        for frame in range(self.total_frames):
            gt_data = self.gt_trajectory[frame]
            
            # 1. Simulate Camera Tracklets (30 FPS)
            # Introduce occlusion period: Frames 150 to 250 (3.33 seconds)
            if 150 <= frame <= 250:
                camera_tracklets = [] # Person out of frame
            else:
                # Assume a single tracklet for the person
                cam_pos = gt_data['position'] + np.random.normal(0, 0.1, 2)  # Camera noise
                cam_vel = gt_data['velocity'] + np.random.normal(0, 0.05, 2) # Velocity noise
                
                camera_tracklets = [{
                    'id': 101,  # Fixed ID for the person
                    'position': cam_pos,
                    'velocity': cam_vel
                }]
            
            # 2. Simulate UWB Measurement (3 FPS)
            uwb_measurement = None
            if frame % self.uwb_interval == 0:
                uwb_pos = gt_data['position'] + np.random.normal(0, 0.5, 2)  # UWB noise (higher)
                uwb_measurement = {
                    'position': uwb_pos,
                    'frame': frame,
                    'timestamp': gt_data['timestamp']
                }
            
            yield {
                'frame': frame,
                'timestamp': gt_data['timestamp'],
                'camera_tracklets': camera_tracklets,
                'uwb_measurement': uwb_measurement
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
    
    def get_initial_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get initial position and velocity."""
        initial_data = self.gt_trajectory[0]
        return initial_data['position'], initial_data['velocity']


def create_streamer(duration: float = 20.0) -> VehicleDataStreamer:
    """Helper function to create a streamer instance."""
    return VehicleDataStreamer(total_duration_sec=duration)
