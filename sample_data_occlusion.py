"""
Sample Data Generation for Occlusion and Re-identification Testing
===================================================================

This module generates synthetic camera tracklets and UWB measurements with
realistic occlusion scenarios to test the re-identification logic.

Scenarios:
1. Person walks through frame, exits, and re-enters from same side
2. Person walks through frame, exits, and re-enters from different side
3. Multiple persons with overlapping occlusion periods
4. Person exits and re-enters with significant position change
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class UWBMeasurement:
    """UWB measurement at a specific frame."""
    frame: int
    position: np.ndarray  # [x, y]
    timestamp: float


@dataclass
class CameraTracklet:
    """Camera tracklet (continuous track of a person)."""
    tracklet_id: int
    frames: List[int]
    positions: List[np.ndarray]  # List of [x, y] positions
    velocities: List[np.ndarray]  # List of [v_x, v_y] velocities
    
    def get_position_at_frame(self, frame):
        """Get position at a specific frame."""
        if frame in self.frames:
            idx = self.frames.index(frame)
            return self.positions[idx]
        return None
    
    def get_velocity_at_frame(self, frame):
        """Get velocity at a specific frame."""
        if frame in self.frames:
            idx = self.frames.index(frame)
            return self.velocities[idx]
        return None


class OcclusionScenarioGenerator:
    """Generates synthetic data with occlusion scenarios."""
    
    def __init__(self, total_frames=600, fps=30, dt=None):
        """
        Initialize the scenario generator.
        
        Args:
            total_frames: Total number of frames to generate
            fps: Frames per second
            dt: Time step (if None, calculated from fps)
        """
        self.total_frames = total_frames
        self.fps = fps
        self.dt = dt if dt is not None else 1.0 / fps
        self.uwb_measurements = []
        self.camera_tracklets = []
        self.tracklet_counter = 0
    
    def generate_scenario_1_simple_occlusion(self):
        """
        Scenario 1: Person walks through frame, exits, and re-enters from same side.
        
        Timeline:
        - Frames 0-100: Person enters and walks across frame (tracklet 1)
        - Frames 100-200: Person out of frame (UWB continues, no camera)
        - Frames 200-300: Person re-enters and walks across frame (tracklet 2, should be re-identified as same person)
        """
        print("\n=== Scenario 1: Simple Occlusion (Same Side Re-entry) ===")
        
        # Generate UWB measurements (3 FPS)
        uwb_frames = list(range(0, self.total_frames, 10))  # 3 FPS equivalent
        
        # Person trajectory (constant velocity)
        for frame in uwb_frames:
            if frame < 100:
                # Entering frame: moving right
                x = 10.0 + (frame / 100.0) * 20.0
                y = 50.0 + np.random.normal(0, 0.5)
            elif frame < 200:
                # Out of frame: continuing motion
                x = 30.0 + ((frame - 100) / 100.0) * 20.0
                y = 50.0 + np.random.normal(0, 0.5)
            else:
                # Re-entering: continuing motion
                x = 50.0 + ((frame - 200) / 100.0) * 10.0
                y = 50.0 + np.random.normal(0, 0.5)
            
            self.uwb_measurements.append(UWBMeasurement(
                frame=frame,
                position=np.array([x, y]),
                timestamp=frame * self.dt
            ))
        
        # Generate camera tracklets (30 FPS)
        # Tracklet 1: Frames 0-100
        tracklet_1_frames = list(range(0, 101))
        tracklet_1_positions = []
        tracklet_1_velocities = []
        
        for frame in tracklet_1_frames:
            x = 10.0 + (frame / 100.0) * 20.0
            y = 50.0 + np.random.normal(0, 0.3)
            tracklet_1_positions.append(np.array([x, y]))
            
            if frame > 0:
                v_x = (tracklet_1_positions[-1][0] - tracklet_1_positions[-2][0]) / self.dt
                v_y = (tracklet_1_positions[-1][1] - tracklet_1_positions[-2][1]) / self.dt
            else:
                v_x, v_y = 0.2, 0.0
            tracklet_1_velocities.append(np.array([v_x, v_y]))
        
        self.camera_tracklets.append(CameraTracklet(
            tracklet_id=0,
            frames=tracklet_1_frames,
            positions=tracklet_1_positions,
            velocities=tracklet_1_velocities
        ))
        
        # Tracklet 2: Frames 200-300 (re-entry, should be re-identified)
        tracklet_2_frames = list(range(200, 301))
        tracklet_2_positions = []
        tracklet_2_velocities = []
        
        for frame in tracklet_2_frames:
            x = 50.0 + ((frame - 200) / 100.0) * 10.0
            y = 50.0 + np.random.normal(0, 0.3)
            tracklet_2_positions.append(np.array([x, y]))
            
            if frame > 200:
                v_x = (tracklet_2_positions[-1][0] - tracklet_2_positions[-2][0]) / self.dt
                v_y = (tracklet_2_positions[-1][1] - tracklet_2_positions[-2][1]) / self.dt
            else:
                v_x, v_y = 0.1, 0.0
            tracklet_2_velocities.append(np.array([v_x, v_y]))
        
        self.camera_tracklets.append(CameraTracklet(
            tracklet_id=1,
            frames=tracklet_2_frames,
            positions=tracklet_2_positions,
            velocities=tracklet_2_velocities
        ))
        
        print(f"Generated UWB measurements: {len(self.uwb_measurements)} (3 FPS)")
        print(f"Generated camera tracklets: {len(self.camera_tracklets)}")
        print(f"  - Tracklet 0: Frames 0-100 (entry)")
        print(f"  - Tracklet 1: Frames 200-300 (re-entry, should be re-identified)")
        print(f"Occlusion period: Frames 100-200 (100 frames, ~3.33 seconds)")
    
    def generate_scenario_2_multiple_persons(self):
        """
        Scenario 2: Multiple persons with overlapping occlusion periods.
        
        Timeline:
        - Person A: Frames 0-150 visible, 150-250 occluded, 250-350 re-enters
        - Person B: Frames 50-200 visible, 200-300 occluded, 300-400 re-enters
        """
        print("\n=== Scenario 2: Multiple Persons with Overlapping Occlusions ===")
        
        self.uwb_measurements = []
        self.camera_tracklets = []
        
        # Person A: UWB measurements
        for frame in range(0, self.total_frames, 10):  # 3 FPS
            if frame < 150:
                x = 20.0 + (frame / 150.0) * 30.0
                y = 40.0
            elif frame < 250:
                # Occluded, but UWB continues
                x = 50.0 + ((frame - 150) / 100.0) * 20.0
                y = 40.0
            else:
                x = 70.0 + ((frame - 250) / 100.0) * 10.0
                y = 40.0
            
            self.uwb_measurements.append(UWBMeasurement(
                frame=frame,
                position=np.array([x, y]),
                timestamp=frame * self.dt
            ))
        
        # Person A: Camera tracklet 1 (visible 0-150)
        tracklet_frames = list(range(0, 151))
        tracklet_positions = [np.array([20.0 + (f / 150.0) * 30.0, 40.0]) for f in tracklet_frames]
        tracklet_velocities = [np.array([0.2, 0.0]) for _ in tracklet_frames]
        
        self.camera_tracklets.append(CameraTracklet(
            tracklet_id=0,
            frames=tracklet_frames,
            positions=tracklet_positions,
            velocities=tracklet_velocities
        ))
        
        # Person A: Camera tracklet 2 (re-entry 250-350)
        tracklet_frames = list(range(250, 351))
        tracklet_positions = [np.array([70.0 + ((f - 250) / 100.0) * 10.0, 40.0]) for f in tracklet_frames]
        tracklet_velocities = [np.array([0.1, 0.0]) for _ in tracklet_frames]
        
        self.camera_tracklets.append(CameraTracklet(
            tracklet_id=1,
            frames=tracklet_frames,
            positions=tracklet_positions,
            velocities=tracklet_velocities
        ))
        
        # Person B: Camera tracklet 1 (visible 50-200)
        tracklet_frames = list(range(50, 201))
        tracklet_positions = [np.array([10.0 + ((f - 50) / 150.0) * 25.0, 60.0]) for f in tracklet_frames]
        tracklet_velocities = [np.array([0.167, 0.0]) for _ in tracklet_frames]
        
        self.camera_tracklets.append(CameraTracklet(
            tracklet_id=2,
            frames=tracklet_frames,
            positions=tracklet_positions,
            velocities=tracklet_velocities
        ))
        
        # Person B: Camera tracklet 2 (re-entry 300-400)
        tracklet_frames = list(range(300, 401))
        tracklet_positions = [np.array([35.0 + ((f - 300) / 100.0) * 15.0, 60.0]) for f in tracklet_frames]
        tracklet_velocities = [np.array([0.15, 0.0]) for _ in tracklet_frames]
        
        self.camera_tracklets.append(CameraTracklet(
            tracklet_id=3,
            frames=tracklet_frames,
            positions=tracklet_positions,
            velocities=tracklet_velocities
        ))
        
        print(f"Generated UWB measurements: {len(self.uwb_measurements)}")
        print(f"Generated camera tracklets: {len(self.camera_tracklets)}")
        print(f"  - Person A: Tracklet 0 (0-150), Tracklet 1 (250-350)")
        print(f"  - Person B: Tracklet 2 (50-200), Tracklet 3 (300-400)")
    
    def generate_scenario_3_long_occlusion(self):
        """
        Scenario 3: Person with long occlusion period (tests hypothesis expiration).
        
        Timeline:
        - Frames 0-100: Person visible
        - Frames 100-400: Person occluded (long period)
        - Frames 400-500: Person re-enters (should NOT be re-identified due to expiration)
        """
        print("\n=== Scenario 3: Long Occlusion (Hypothesis Expiration) ===")
        
        self.uwb_measurements = []
        self.camera_tracklets = []
        
        # UWB measurements throughout
        for frame in range(0, self.total_frames, 10):  # 3 FPS
            if frame < 100:
                x = 30.0 + (frame / 100.0) * 20.0
                y = 50.0
            elif frame < 400:
                # Occluded, UWB continues
                x = 50.0 + ((frame - 100) / 300.0) * 30.0
                y = 50.0
            else:
                x = 80.0 + ((frame - 400) / 100.0) * 10.0
                y = 50.0
            
            self.uwb_measurements.append(UWBMeasurement(
                frame=frame,
                position=np.array([x, y]),
                timestamp=frame * self.dt
            ))
        
        # Camera tracklet 1 (visible 0-100)
        tracklet_frames = list(range(0, 101))
        tracklet_positions = [np.array([30.0 + (f / 100.0) * 20.0, 50.0]) for f in tracklet_frames]
        tracklet_velocities = [np.array([0.2, 0.0]) for _ in tracklet_frames]
        
        self.camera_tracklets.append(CameraTracklet(
            tracklet_id=0,
            frames=tracklet_frames,
            positions=tracklet_positions,
            velocities=tracklet_velocities
        ))
        
        # Camera tracklet 2 (re-entry 400-500, long occlusion)
        tracklet_frames = list(range(400, 501))
        tracklet_positions = [np.array([80.0 + ((f - 400) / 100.0) * 10.0, 50.0]) for f in tracklet_frames]
        tracklet_velocities = [np.array([0.1, 0.0]) for _ in tracklet_frames]
        
        self.camera_tracklets.append(CameraTracklet(
            tracklet_id=1,
            frames=tracklet_frames,
            positions=tracklet_positions,
            velocities=tracklet_velocities
        ))
        
        print(f"Generated UWB measurements: {len(self.uwb_measurements)}")
        print(f"Generated camera tracklets: {len(self.camera_tracklets)}")
        print(f"  - Tracklet 0: Frames 0-100 (visible)")
        print(f"  - Tracklet 1: Frames 400-500 (re-entry after long occlusion)")
        print(f"Occlusion period: Frames 100-400 (300 frames, 10 seconds)")
    
    def get_uwb_measurement_at_frame(self, frame):
        """Get UWB measurement at a specific frame."""
        for measurement in self.uwb_measurements:
            if measurement.frame == frame:
                return measurement
        return None
    
    def get_camera_tracklet_at_frame(self, frame):
        """Get camera tracklet(s) at a specific frame."""
        tracklets_at_frame = []
        for tracklet in self.camera_tracklets:
            if frame in tracklet.frames:
                tracklets_at_frame.append(tracklet)
        return tracklets_at_frame
    
    def get_all_tracklets_in_range(self, start_frame, end_frame):
        """Get all tracklets that exist in a frame range."""
        tracklets_in_range = []
        for tracklet in self.camera_tracklets:
            if any(start_frame <= f <= end_frame for f in tracklet.frames):
                tracklets_in_range.append(tracklet)
        return tracklets_in_range
    
    def print_timeline(self):
        """Print a timeline of all tracklets and measurements."""
        print("\n=== Timeline ===")
        print(f"Total frames: {self.total_frames}")
        print(f"Duration: {self.total_frames * self.dt:.2f} seconds")
        print(f"\nCamera Tracklets:")
        for tracklet in self.camera_tracklets:
            start_frame = min(tracklet.frames)
            end_frame = max(tracklet.frames)
            duration = (end_frame - start_frame) * self.dt
            print(f"  Tracklet {tracklet.tracklet_id}: Frames {start_frame}-{end_frame} ({duration:.2f}s)")
        
        print(f"\nUWB Measurements: {len(self.uwb_measurements)} (3 FPS)")


def create_test_scenarios():
    """Create all test scenarios and return them."""
    scenarios = {}
    
    # Scenario 1: Simple occlusion
    gen1 = OcclusionScenarioGenerator(total_frames=300)
    gen1.generate_scenario_1_simple_occlusion()
    scenarios['simple_occlusion'] = gen1
    
    # Scenario 2: Multiple persons
    gen2 = OcclusionScenarioGenerator(total_frames=400)
    gen2.generate_scenario_2_multiple_persons()
    scenarios['multiple_persons'] = gen2
    
    # Scenario 3: Long occlusion
    gen3 = OcclusionScenarioGenerator(total_frames=500)
    gen3.generate_scenario_3_long_occlusion()
    scenarios['long_occlusion'] = gen3
    
    return scenarios
