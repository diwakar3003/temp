"""
Main Integration Script: Occlusion and Re-identification Testing
=================================================================

This script integrates the UKF with occlusion handling and the re-identification
logic to demonstrate how the system handles persons going out of frame and
re-entering.
"""

import numpy as np
from uwb_ukf_occlusion import UWBUKFOcclusion, OcclusionTracker
from reidentification_logic import HungarianReidentificationMatcher
from sample_data_occlusion import create_test_scenarios


class OptinCameraOcclusionSystem:
    """
    Complete Opt-in Camera system with occlusion and re-identification handling.
    """
    
    def __init__(self, dt=1/30.0, max_occlusion_time=5.0):
        """
        Initialize the system.
        
        Args:
            dt: Time step (seconds)
            max_occlusion_time: Maximum occlusion duration to track (seconds)
        """
        self.dt = dt
        self.max_occlusion_time = max_occlusion_time
        
        # UKF for tracking
        self.ukf = UWBUKFOcclusion(dt=dt)
        
        # Re-identification matcher
        self.reidentification_matcher = HungarianReidentificationMatcher(
            max_occlusion_time=max_occlusion_time,
            spatial_threshold=3.0,
            temporal_threshold=2.0,
            confidence_threshold=0.6
        )
        
        # Tracking state
        self.current_tracklet_id = None
        self.reidentified_pairs = {}  # Maps old tracklet IDs to new ones
        self.frame_history = []
    
    def process_frame(self, frame_number, uwb_measurement=None, camera_tracklets=None):
        """
        Process a single frame.
        
        Args:
            frame_number: Current frame number
            uwb_measurement: UWB measurement [x, y] or None
            camera_tracklets: List of camera tracklets at this frame or None
            
        Returns:
            Dictionary with processing results
        """
        result = {
            'frame': frame_number,
            'ukf_position': None,
            'ukf_uncertainty': None,
            'is_occluded': False,
            'reidentification_match': None,
            'reidentification_confidence': 0.0,
            'camera_tracklet_id': None
        }
        
        # Step 1: UKF Prediction (always runs)
        self.ukf.predict(frame_number)
        result['ukf_position'] = self.ukf.get_position()
        result['ukf_uncertainty'] = self.ukf.get_uncertainty()
        result['is_occluded'] = self.ukf.is_occluded
        
        # Step 2: Handle camera tracklets
        if camera_tracklets is None or len(camera_tracklets) == 0:
            # No camera tracklet at this frame - mark as occluded
            if not self.ukf.is_occluded:
                self.ukf.start_occlusion(frame_number)
                self.reidentification_matcher.register_missing_person(
                    person_id=self.current_tracklet_id,
                    disappearance_frame=frame_number,
                    last_position=self.ukf.get_position(),
                    last_velocity=self.ukf.get_velocity(),
                    last_covariance=self.ukf.get_position_covariance()
                )
        else:
            # Camera tracklet available
            tracklet = camera_tracklets[0]  # Take first tracklet
            
            # Check if this is a re-identification
            if self.ukf.is_occluded:
                # Person was occluded, now re-entering
                matched_person_id, confidence = self.reidentification_matcher.match_new_tracklet(
                    new_tracklet_id=tracklet.tracklet_id,
                    new_position=tracklet.get_position_at_frame(frame_number),
                    new_velocity=tracklet.get_velocity_at_frame(frame_number),
                    current_frame=frame_number,
                    dt=self.dt
                )
                
                if matched_person_id is not None:
                    result['reidentification_match'] = matched_person_id
                    result['reidentification_confidence'] = confidence
                    self.reidentified_pairs[matched_person_id] = tracklet.tracklet_id
                    print(f"[Frame {frame_number}] RE-IDENTIFICATION SUCCESS!")
                    print(f"  Old tracklet: {matched_person_id}, New tracklet: {tracklet.tracklet_id}")
                    print(f"  Confidence: {confidence:.4f}")
                else:
                    print(f"[Frame {frame_number}] New tracklet {tracklet.tracklet_id} could not be re-identified")
            
            # Update UKF with camera measurement
            position = tracklet.get_position_at_frame(frame_number)
            if position is not None:
                self.ukf.update(position, frame_number)
                self.current_tracklet_id = tracklet.tracklet_id
                result['camera_tracklet_id'] = tracklet.tracklet_id
                
                # Mark as visible if was occluded
                if self.ukf.is_occluded:
                    self.ukf.end_occlusion(frame_number)
        
        # Step 3: Clean up expired re-identification hypotheses
        self.reidentification_matcher.remove_expired_hypotheses(frame_number, self.dt)
        
        # Store frame history
        self.frame_history.append(result)
        
        return result
    
    def process_scenario(self, scenario_generator):
        """
        Process an entire scenario.
        
        Args:
            scenario_generator: OcclusionScenarioGenerator instance
        """
        print(f"\n{'='*70}")
        print(f"Processing Scenario")
        print(f"{'='*70}")
        
        scenario_generator.print_timeline()
        
        # Process each frame
        for frame in range(scenario_generator.total_frames):
            # Get UWB measurement (if available)
            uwb_measurement = scenario_generator.get_uwb_measurement_at_frame(frame)
            
            # Get camera tracklets (if available)
            camera_tracklets = scenario_generator.get_camera_tracklet_at_frame(frame)
            
            # Process frame
            result = self.process_frame(
                frame_number=frame,
                uwb_measurement=uwb_measurement,
                camera_tracklets=camera_tracklets
            )
        
        # Print summary
        self.print_summary(scenario_generator)
    
    def print_summary(self, scenario_generator):
        """Print a summary of the processing results."""
        print(f"\n{'='*70}")
        print(f"Summary")
        print(f"{'='*70}")
        
        # Count occlusion periods
        occlusion_periods = 0
        current_occlusion = False
        occlusion_start = None
        
        for result in self.frame_history:
            if result['is_occluded'] and not current_occlusion:
                occlusion_periods += 1
                current_occlusion = True
                occlusion_start = result['frame']
            elif not result['is_occluded'] and current_occlusion:
                current_occlusion = False
        
        print(f"\nOcclusion Statistics:")
        print(f"  Total occlusion periods: {occlusion_periods}")
        print(f"  Total frames processed: {len(self.frame_history)}")
        
        # Re-identification statistics
        reidentification_report = self.reidentification_matcher.get_reidentification_report()
        print(f"\nRe-identification Statistics:")
        print(f"  Total re-identification matches: {reidentification_report['total_matches']}")
        print(f"  Matches: {reidentification_report['matches']}")
        
        if reidentification_report['scores']:
            print(f"\nRe-identification Scores:")
            for (old_id, new_id), score in reidentification_report['scores'].items():
                print(f"  Old tracklet {old_id} -> New tracklet {new_id}: {score:.4f}")
        
        # UKF statistics
        print(f"\nUKF Statistics:")
        print(f"  Final position: {self.ukf.get_position()}")
        print(f"  Final velocity: {self.ukf.get_velocity()}")
        print(f"  Final uncertainty: {self.ukf.get_uncertainty():.4f}")
        
        # Camera tracklet statistics
        print(f"\nCamera Tracklet Statistics:")
        for tracklet in scenario_generator.camera_tracklets:
            start_frame = min(tracklet.frames)
            end_frame = max(tracklet.frames)
            duration = (end_frame - start_frame) * self.dt
            print(f"  Tracklet {tracklet.tracklet_id}: Frames {start_frame}-{end_frame} ({duration:.2f}s)")


def main():
    """Main function to run all test scenarios."""
    print("\n" + "="*70)
    print("Opt-in Camera System: Occlusion and Re-identification Testing")
    print("="*70)
    
    # Create test scenarios
    scenarios = create_test_scenarios()
    
    # Test Scenario 1: Simple Occlusion
    print("\n" + "="*70)
    print("TEST SCENARIO 1: Simple Occlusion (Same Side Re-entry)")
    print("="*70)
    system1 = OptinCameraOcclusionSystem(dt=1/30.0, max_occlusion_time=5.0)
    system1.process_scenario(scenarios['simple_occlusion'])
    
    # Test Scenario 2: Multiple Persons
    print("\n" + "="*70)
    print("TEST SCENARIO 2: Multiple Persons with Overlapping Occlusions")
    print("="*70)
    system2 = OptinCameraOcclusionSystem(dt=1/30.0, max_occlusion_time=5.0)
    system2.process_scenario(scenarios['multiple_persons'])
    
    # Test Scenario 3: Long Occlusion
    print("\n" + "="*70)
    print("TEST SCENARIO 3: Long Occlusion (Hypothesis Expiration)")
    print("="*70)
    system3 = OptinCameraOcclusionSystem(dt=1/30.0, max_occlusion_time=5.0)
    system3.process_scenario(scenarios['long_occlusion'])
    
    print("\n" + "="*70)
    print("All scenarios completed!")
    print("="*70)


if __name__ == "__main__":
    main()
