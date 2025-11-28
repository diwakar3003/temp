"""
Streaming UKF for Real-Time Vehicle Inference
==============================================

This module implements a 2D UKF optimized for real-time streaming data
from vehicles. It handles asynchronous data streams (3 FPS UWB, 30 FPS camera)
with buffer management and frame synchronization.

Key Features:
- Circular buffers for efficient memory usage
- Frame synchronization between UWB and camera
- Latency tracking and performance metrics
- Vehicle-specific motion constraints
"""

import numpy as np
from collections import deque
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import time
from dataclasses import dataclass
from typing import Optional, Dict, Tuple


@dataclass
class StreamingMetrics:
    """Performance metrics for streaming inference."""
    frame_number: int
    timestamp: float
    ukf_latency_ms: float
    buffer_size: int
    uwb_buffer_size: int
    camera_buffer_size: int
    fps: float
    processing_time_ms: float


class StreamingUWBUKF2D:
    """
    Streaming 2D UKF optimized for real-time vehicle inference.
    
    Handles asynchronous UWB (3 FPS) and camera (30 FPS) data streams
    with efficient buffer management.
    """
    
    def __init__(self, 
                 dt=1/30.0,
                 process_noise_std=0.5,
                 measurement_noise_std=2.0,
                 buffer_size=100,
                 max_uwb_age_frames=15):
        """
        Initialize streaming UKF.
        
        Args:
            dt: Time step (default: 1/30 seconds for 30 FPS)
            process_noise_std: Process noise standard deviation
            measurement_noise_std: Measurement noise standard deviation
            buffer_size: Maximum buffer size for history
            max_uwb_age_frames: Maximum age of UWB measurements (frames)
        """
        self.dt = dt
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std
        self.buffer_size = buffer_size
        self.max_uwb_age_frames = max_uwb_age_frames
        
        # State dimension: [x, v_x, y, v_y]
        self.state_dim = 4
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
            hx=self._h,
            fx=self._f,
            points=self.sigma_points
        )
        
        # Set noise covariances
        q = self.process_noise_std ** 2
        self.ukf.Q = np.eye(self.state_dim) * q
        r = self.measurement_noise_std ** 2
        self.ukf.R = np.eye(self.measurement_dim) * r
        self.ukf.P = np.eye(self.state_dim) * 1.0
        self.ukf.x = np.array([0.0, 0.0, 0.0, 0.0])
        
        # Circular buffers for streaming data
        self.state_buffer = deque(maxlen=buffer_size)
        self.covariance_buffer = deque(maxlen=buffer_size)
        self.measurement_buffer = deque(maxlen=buffer_size)
        self.timestamp_buffer = deque(maxlen=buffer_size)
        self.frame_buffer = deque(maxlen=buffer_size)
        
        # UWB measurement buffer (separate, for asynchronous updates)
        self.uwb_buffer = deque(maxlen=buffer_size)
        
        # Tracking state
        self.frame_count = 0
        self.last_uwb_frame = -1
        self.start_time = time.time()
        
        # Performance metrics
        self.metrics_history = deque(maxlen=buffer_size)
        self.processing_times = deque(maxlen=100)
    
    def _f(self, x, dt):
        """State transition function (Constant Velocity)."""
        F = np.array([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1]
        ])
        return F @ x
    
    def _h(self, x):
        """Measurement function (direct position observation)."""
        return np.array([x[0], x[2]])
    
    def predict(self, frame_number: int, timestamp: float = None) -> StreamingMetrics:
        """
        Predict step for streaming data.
        
        Args:
            frame_number: Current frame number
            timestamp: Current timestamp (if None, calculated from frame_number)
            
        Returns:
            StreamingMetrics with performance information
        """
        if timestamp is None:
            timestamp = frame_number * self.dt
        
        start_time = time.time()
        
        # Predict
        self.ukf.predict()
        
        # Store in buffers
        self.state_buffer.append(self.ukf.x.copy())
        self.covariance_buffer.append(self.ukf.P.copy())
        self.frame_buffer.append(frame_number)
        self.timestamp_buffer.append(timestamp)
        
        # Calculate metrics
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        self.processing_times.append(processing_time)
        
        # Update frame count
        self.frame_count = frame_number
        
        # Create metrics
        metrics = StreamingMetrics(
            frame_number=frame_number,
            timestamp=timestamp,
            ukf_latency_ms=processing_time,
            buffer_size=len(self.state_buffer),
            uwb_buffer_size=len(self.uwb_buffer),
            camera_buffer_size=len(self.measurement_buffer),
            fps=1.0 / self.dt,
            processing_time_ms=processing_time
        )
        
        self.metrics_history.append(metrics)
        
        return metrics
    
    def update_camera(self, measurement: np.ndarray, frame_number: int, 
                     timestamp: float = None) -> StreamingMetrics:
        """
        Update with camera measurement (30 FPS).
        
        Args:
            measurement: Camera position [x, y]
            frame_number: Current frame number
            timestamp: Current timestamp
            
        Returns:
            StreamingMetrics with performance information
        """
        if timestamp is None:
            timestamp = frame_number * self.dt
        
        start_time = time.time()
        
        # Update UKF
        self.ukf.update(measurement)
        
        # Store measurement
        self.measurement_buffer.append(measurement.copy())
        
        # Calculate metrics
        processing_time = (time.time() - start_time) * 1000
        self.processing_times.append(processing_time)
        
        metrics = StreamingMetrics(
            frame_number=frame_number,
            timestamp=timestamp,
            ukf_latency_ms=processing_time,
            buffer_size=len(self.state_buffer),
            uwb_buffer_size=len(self.uwb_buffer),
            camera_buffer_size=len(self.measurement_buffer),
            fps=1.0 / self.dt,
            processing_time_ms=processing_time
        )
        
        self.metrics_history.append(metrics)
        
        return metrics
    
    def update_uwb(self, measurement: np.ndarray, frame_number: int,
                  timestamp: float = None) -> bool:
        """
        Buffer UWB measurement (3 FPS) for asynchronous processing.
        
        Args:
            measurement: UWB position [x, y]
            frame_number: Frame number when measurement arrived
            timestamp: Timestamp of measurement
            
        Returns:
            True if measurement was processed, False if buffered
        """
        if timestamp is None:
            timestamp = frame_number * self.dt
        
        # Store in UWB buffer
        self.uwb_buffer.append({
            'measurement': measurement.copy(),
            'frame': frame_number,
            'timestamp': timestamp,
            'age': 0
        })
        
        # Check if we should process this UWB measurement
        # Process if enough frames have passed since last UWB update
        if frame_number - self.last_uwb_frame >= 10:  # ~3 FPS at 30 FPS camera
            # Get the most recent UWB measurement
            if len(self.uwb_buffer) > 0:
                uwb_data = self.uwb_buffer[-1]
                self.update_camera(uwb_data['measurement'], frame_number, timestamp)
                self.last_uwb_frame = frame_number
                return True
        
        return False
    
    def get_state(self) -> np.ndarray:
        """Get current state [x, v_x, y, v_y]."""
        return self.ukf.x.copy()
    
    def get_position(self) -> np.ndarray:
        """Get current position [x, y]."""
        return np.array([self.ukf.x[0], self.ukf.x[2]])
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity [v_x, v_y]."""
        return np.array([self.ukf.x[1], self.ukf.x[3]])
    
    def get_covariance(self) -> np.ndarray:
        """Get current covariance matrix."""
        return self.ukf.P.copy()
    
    def get_position_covariance(self) -> np.ndarray:
        """Get position covariance (2x2)."""
        return self.ukf.P[np.ix_([0, 2], [0, 2])]
    
    def get_uncertainty(self) -> float:
        """Get position uncertainty (trace of covariance)."""
        pos_cov = self.get_position_covariance()
        return np.trace(pos_cov)
    
    def get_average_processing_time(self) -> float:
        """Get average processing time in milliseconds."""
        if len(self.processing_times) == 0:
            return 0.0
        return np.mean(self.processing_times)
    
    def get_max_processing_time(self) -> float:
        """Get maximum processing time in milliseconds."""
        if len(self.processing_times) == 0:
            return 0.0
        return np.max(self.processing_times)
    
    def initialize_state(self, initial_position: np.ndarray, 
                        initial_velocity: np.ndarray = None):
        """Initialize UKF state."""
        if initial_velocity is None:
            initial_velocity = np.array([0.0, 0.0])
        
        self.ukf.x = np.array([
            initial_position[0],
            initial_velocity[0],
            initial_position[1],
            initial_velocity[1]
        ])
        
        self.ukf.P = np.eye(self.state_dim) * 1.0
    
    def get_buffer_status(self) -> Dict:
        """Get current buffer status."""
        return {
            'state_buffer': len(self.state_buffer),
            'measurement_buffer': len(self.measurement_buffer),
            'uwb_buffer': len(self.uwb_buffer),
            'timestamp_buffer': len(self.timestamp_buffer),
            'frame_buffer': len(self.frame_buffer),
            'max_buffer_size': self.buffer_size
        }
    
    def get_latest_metrics(self) -> Optional[StreamingMetrics]:
        """Get latest metrics."""
        if len(self.metrics_history) > 0:
            return self.metrics_history[-1]
        return None
    
    def reset(self):
        """Reset all buffers and state."""
        self.state_buffer.clear()
        self.covariance_buffer.clear()
        self.measurement_buffer.clear()
        self.timestamp_buffer.clear()
        self.frame_buffer.clear()
        self.uwb_buffer.clear()
        self.metrics_history.clear()
        self.processing_times.clear()
        self.frame_count = 0
        self.last_uwb_frame = -1
        self.start_time = time.time()


class StreamingDataBuffer:
    """
    Manages streaming data buffers for vehicle inference.
    Handles synchronization between UWB and camera streams.
    """
    
    def __init__(self, buffer_size=1000):
        """
        Initialize streaming data buffer.
        
        Args:
            buffer_size: Maximum buffer size
        """
        self.buffer_size = buffer_size
        self.camera_buffer = deque(maxlen=buffer_size)
        self.uwb_buffer = deque(maxlen=buffer_size)
        self.frame_counter = 0
        self.last_sync_frame = 0
    
    def add_camera_frame(self, frame_number: int, tracklets: list, 
                        timestamp: float = None) -> bool:
        """
        Add camera frame data to buffer.
        
        Args:
            frame_number: Frame number
            tracklets: List of tracklets in this frame
            timestamp: Timestamp of frame
            
        Returns:
            True if frame was added
        """
        if timestamp is None:
            timestamp = frame_number / 30.0  # Assume 30 FPS
        
        self.camera_buffer.append({
            'frame': frame_number,
            'tracklets': tracklets,
            'timestamp': timestamp
        })
        
        return True
    
    def add_uwb_measurement(self, frame_number: int, position: np.ndarray,
                          timestamp: float = None) -> bool:
        """
        Add UWB measurement to buffer.
        
        Args:
            frame_number: Frame number
            position: UWB position [x, y]
            timestamp: Timestamp of measurement
            
        Returns:
            True if measurement was added
        """
        if timestamp is None:
            timestamp = frame_number / 30.0
        
        self.uwb_buffer.append({
            'frame': frame_number,
            'position': position.copy(),
            'timestamp': timestamp
        })
        
        return True
    
    def get_camera_frame(self, frame_number: int) -> Optional[Dict]:
        """Get camera frame from buffer."""
        for item in self.camera_buffer:
            if item['frame'] == frame_number:
                return item
        return None
    
    def get_uwb_measurement(self, frame_number: int) -> Optional[Dict]:
        """Get UWB measurement from buffer."""
        for item in self.uwb_buffer:
            if item['frame'] == frame_number:
                return item
        return None
    
    def get_latest_camera_frame(self) -> Optional[Dict]:
        """Get latest camera frame."""
        if len(self.camera_buffer) > 0:
            return self.camera_buffer[-1]
        return None
    
    def get_latest_uwb_measurement(self) -> Optional[Dict]:
        """Get latest UWB measurement."""
        if len(self.uwb_buffer) > 0:
            return self.uwb_buffer[-1]
        return None
    
    def get_buffer_status(self) -> Dict:
        """Get buffer status."""
        return {
            'camera_buffer_size': len(self.camera_buffer),
            'uwb_buffer_size': len(self.uwb_buffer),
            'max_buffer_size': self.buffer_size,
            'frame_counter': self.frame_counter
        }
    
    def clear(self):
        """Clear all buffers."""
        self.camera_buffer.clear()
        self.uwb_buffer.clear()
        self.frame_counter = 0
        self.last_sync_frame = 0
