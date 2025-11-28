"""
Real-Time Inference Pipeline for Vehicle-Mounted Opt-in Camera
=============================================================

This script integrates the streaming UKF, real-time trajectory matching, and
simulated data streamer to demonstrate a complete real-time inference pipeline
for a vehicle-mounted Opt-in Camera system.

It includes performance monitoring to ensure the system meets real-time constraints.
"""

import numpy as np
import time
from uwb_ukf_2d_streaming import StreamingUWBUKF2D
from trajectory_matching_2d_streaming import RealTimeTrajectoryMatcher
from vehicle_data_streamer import create_streamer
from typing import Dict, Any, List


class RealTimeInferencePipeline:
    """
    Manages the entire real-time inference process.
    """
    
    def __init__(self, duration_sec: float = 20.0):
        """
        Initialize the pipeline components.
        
        Args:
            duration_sec: Total duration of the simulation
        """
        self.streamer = create_streamer(duration=duration_sec)
        self.dt = self.streamer.get_dt()
        self.fps = self.streamer.get_fps()
        
        # Initialize UKF
        self.ukf = StreamingUWBUKF2D(dt=self.dt, buffer_size=self.fps * 5) # 5 seconds buffer
        
        # Initialize Matcher (runs LAP every 3 frames for 10 FPS matching rate)
        self.matcher = RealTimeTrajectoryMatcher(
            dt=self.dt,
            matching_rate_fps=10,
            similarity_window_frames=self.fps * 1, # 1 second window
            cost_threshold=1.5 # Adjusted for real-time noise
        )
        
        # Initialize UKF state
        initial_pos, initial_vel = self.streamer.get_initial_state()
        self.ukf.initialize_state(initial_pos, initial_vel)
        
        # Performance tracking
        self.total_processing_time = 0.0
        self.total_frames = 0
        self.start_time = time.time()
    
    def run_inference(self):
        """
        Runs the simulated real-time inference loop.
        """
        print("="*80)
        print(f"Starting Real-Time Inference Simulation")
        print(f"Camera Rate: {self.fps} FPS | UWB Rate: {self.streamer.get_uwb_rate()} FPS")
        print(f"Matching Rate: 10 FPS | Similarity Window: 1.0 sec")
        print("="*80)
        
        for data in self.streamer.stream_data():
            frame = data['frame']
            timestamp = data['timestamp']
            camera_tracklets = data['camera_tracklets']
            uwb_measurement = data['uwb_measurement']
            
            frame_start_time = time.time()
            
            # --- 1. UKF Prediction ---
            self.ukf.predict(frame, timestamp)
            
            # --- 2. UKF Update (Asynchronous UWB) ---
            
            # Check for camera tracklet presence
            is_visible = len(camera_tracklets) > 0
            
            if is_visible:
                # Update UKF with camera tracklet position (simulating UWB-Camera fusion)
                tracklet_pos = camera_tracklets[0]['position']
                self.ukf.update_camera(tracklet_pos, frame, timestamp)
            else:
                # Person is out of frame (occluded). UKF continues to predict.
                # We log this event and ensure no assignment is made.
                if frame == 151: # Log start of occlusion
                    print(f"--- Occlusion Start at Frame {frame} ---")
                if frame == 250: # Log end of occlusion
                    print(f"--- Occlusion End at Frame {frame} ---")
                
                # Note: The UKF update is skipped, relying on the prediction step.
                # In a real system, we would use the raw UWB measurement here if available.
                # For this simulation, we rely purely on the prediction step.
            
            # --- 3. Trajectory Matching Update ---
            
            # --- 3. Trajectory Matching Update ---
            
            # Update UKF state buffer for matching (always update with predicted/updated state)
            self.matcher.update_ukf_state(
                frame=frame,
                position=self.ukf.get_position(),
                covariance=self.ukf.get_position_covariance()
            )
            
            # Update camera tracklet buffer for matching
            cam_data_for_matcher = [
                {'id': t['id'], 'position': t['position']} for t in camera_tracklets
            ]
            self.matcher.update_camera_tracklets(frame, cam_data_for_matcher)
            
            # --- 4. Real-Time Assignment (Runs at 10 FPS) ---
            assignments = self.matcher.run_lap_assignment(frame)
            
            # --- 5. Performance Monitoring and Logging ---
            frame_end_time = time.time()
            frame_processing_time = (frame_end_time - frame_start_time) * 1000 # ms
            self.total_processing_time += frame_processing_time
            self.total_frames += 1
            
            # Log every 30 frames (1 second)
            if frame % 30 == 0:
                self._log_status(frame, frame_processing_time, assignments, camera_tracklets)
        
        self._print_summary()
    
    def _log_status(self, frame: int, processing_time_ms: float, assignments: Dict[int, int], camera_tracklets: List[Dict]):
        """Logs the current status of the pipeline."""
        
        ukf_metrics = self.ukf.get_latest_metrics()
        
        # Check for assignment
        assignment_str = "None"
        if assignments:
            assignment_str = f"UWB {list(assignments.keys())[0]} -> Cam {list(assignments.values())[0]}"
        
        # Log status, including a marker for occlusion
        occlusion_marker = " (OCCLUDED)" if len(camera_tracklets) == 0 else ""
        
        print(f"Frame {frame:04d} | Time: {ukf_metrics.timestamp:.2f}s | Proc Time: {processing_time_ms:.2f}ms | "
              f"UKF Uncert: {self.ukf.get_uncertainty():.2f} | Assignment: {assignment_str}{occlusion_marker}")
        
        # Check for real-time constraint violation (e.g., processing time > 1/30 sec = 33.33ms)
        if processing_time_ms > 1000 / self.fps:
            print(f"!!! WARNING: Frame {frame} violated real-time constraint ({processing_time_ms:.2f}ms > {1000/self.fps:.2f}ms)")
    
    def _print_summary(self):
        """Prints the final summary of the simulation."""
        
        avg_proc_time = self.total_processing_time / self.total_frames
        max_proc_time = self.ukf.get_max_processing_time()
        
        print("="*80)
        print("Simulation Summary")
        print("="*80)
        print(f"Total Frames Processed: {self.total_frames}")
        print(f"Total Duration: {self.total_frames * self.dt:.2f} seconds")
        print(f"Average Frame Processing Time: {avg_proc_time:.2f} ms")
        print(f"Maximum Frame Processing Time: {max_proc_time:.2f} ms")
        print(f"Real-Time Constraint (30 FPS): {1000/self.fps:.2f} ms")
        
        if avg_proc_time < 1000 / self.fps:
            print("STATUS: Real-time constraint MET (Average)")
        else:
            print("STATUS: Real-time constraint FAILED (Average)")
        
        print("\nFinal Assignments:")
        print(self.matcher.get_current_assignments())
        
        print("\nAssignment History (Last 5):")
        for assignment in list(self.matcher.get_assignment_history())[-5:]:
            print(f"  Frame {assignment['frame']:04d}: UWB {assignment['uwb_id']} -> Cam {assignment['tracklet_id']} (Sim: {assignment['similarity']:.4f})")


def main():
    """Entry point for the real-time inference simulation."""
    pipeline = RealTimeInferencePipeline(duration_sec=10.0) # Run for 10 seconds
    pipeline.run_inference()


if __name__ == "__main__":
    main()
