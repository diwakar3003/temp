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
from multi_person_tracker import MultiPersonTracker
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
        
        # Initialize Multi-Person Tracker
        self.tracker = MultiPersonTracker(dt=self.dt, buffer_size=self.fps * 5) # 5 seconds buffer
        
        # Initialize Matcher (runs LAP every 3 frames for 10 FPS matching rate)
        self.matcher = RealTimeTrajectoryMatcher(
            dt=self.dt,
            matching_rate_fps=10,
            similarity_window_frames=self.fps * 1, # 1 second window
            cost_threshold=5.0 # Increased to resolve "infeasible cost matrix" error and ensure feasibility for multi-person noise
        )
        
        # Initialize Tracker states
        initial_states = self.streamer.get_initial_states()
        self.tracker.initialize_trackers(initial_states)
        
        # Get UWB IDs for reference
        self.uwb_ids = self.streamer.get_uwb_ids()
        
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
            uwb_measurements = data['uwb_measurements'] # Changed to plural
            
            frame_start_time = time.time()
            
            # --- 1. UKF Prediction ---
            self.tracker.predict_all(frame, timestamp)
            
            # --- 2. UKF Update (UWB-Driven) ---
            self.tracker.update_all_uwb(uwb_measurements)
            
            # Log occlusion events for clarity in the output
            # We will only log the occlusion of the first person (UWB ID 101) for simplicity
            is_visible_p1 = any(t['id'] == 1 for t in camera_tracklets) # Tracklet ID 1 corresponds to UWB ID 101
            if not is_visible_p1:
                if frame == 151: # Log start of occlusion for P1
                    print(f"--- Occlusion Start (P1) at Frame {frame} ---")
                if frame == 250: # Log end of occlusion for P1
                    print(f"--- Occlusion End (P1) at Frame {frame} ---")
            
            # --- 3. Trajectory Matching Update ---
            
            # --- 3. Trajectory Matching Update ---
            
            # --- 3. Trajectory Matching Update ---
            
            # Update UKF state buffer for matching (for all UWB tags)
            ukf_states = self.tracker.get_ukf_states_for_matching()
            ukf_metrics = self.tracker.get_ukf_metrics()
            
            # Filter out UKF states with very high uncertainty (e.g., before first UWB update)
            # We use a threshold of 100.0 for the trace of the covariance matrix.
            UNCERTAINTY_THRESHOLD = 100.0
            
            for uwb_id, state in ukf_states.items():
                uncertainty = ukf_metrics.get(uwb_id, {}).get('uncertainty', UNCERTAINTY_THRESHOLD + 1)
                
                if uncertainty < UNCERTAINTY_THRESHOLD:
                    self.matcher.update_ukf_state(
                        frame=frame,
                        uwb_id=uwb_id,
                        position=state['position'],
                        covariance=state['covariance']
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
        
        ukf_metrics = self.tracker.get_ukf_metrics()
        
        # Check for assignment
        assignment_str = "None"
        if assignments:
            assignment_str = ", ".join([f"UWB {u} -> Cam {c}" for u, c in assignments.items()])
        
        # Log status, including a marker for occlusion
        occlusion_marker = " (OCCLUDED)" if len(camera_tracklets) == 0 else ""
        
        # Log uncertainty for UWB ID 101 and 102
        uncert_101 = ukf_metrics.get(101, {}).get('uncertainty', 0.0)
        uncert_102 = ukf_metrics.get(102, {}).get('uncertainty', 0.0)
        
        print(f"Frame {frame:04d} | Time: {ukf_metrics.get(101, {}).get('timestamp', 0.0):.2f}s | Proc Time: {processing_time_ms:.2f}ms | "
              f"Uncert(101/102): {uncert_101:.2f}/{uncert_102:.2f} | Assignment: {assignment_str}{occlusion_marker}")
        
        # Check for real-time constraint violation (e.g., processing time > 1/30 sec = 33.33ms)
        if processing_time_ms > 1000 / self.fps:
            print(f"!!! WARNING: Frame {frame} violated real-time constraint ({processing_time_ms:.2f}ms > {1000/self.fps:.2f}ms)")
    
    def _print_summary(self):
        """Prints the final summary of the simulation."""
        
        avg_proc_time = self.total_processing_time / self.total_frames
        max_proc_time = self.tracker.get_max_processing_time()
        
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
    pipeline = RealTimeInferencePipeline(duration_sec=20.0) # Run for 20 seconds to allow for multi-person scenarios to play out
    pipeline.run_inference()


if __name__ == "__main__":
    main()
