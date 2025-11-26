"""
Re-identification Logic for Occluded Persons
==============================================

This module implements the re-identification logic that matches new camera tracklets
to previously occluded persons. It uses multiple strategies:

1. Mahalanobis Distance: Probabilistic distance based on UKF uncertainty
2. Temporal Proximity: Time gap between disappearance and reappearance
3. Spatial Proximity: Distance between predicted position and new tracklet
4. Velocity Consistency: Check if velocity direction is consistent
5. Confidence Scoring: Combine multiple metrics into a single confidence score

The system maintains a list of "missing" persons (those who disappeared from frame)
and attempts to match new tracklets to them based on these criteria.
"""

import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.optimize import linear_sum_assignment


class ReidentificationMatcher:
    """
    Matches new camera tracklets to previously occluded persons using multiple strategies.
    """
    
    def __init__(self, 
                 max_occlusion_time=5.0,  # Maximum time (seconds) to maintain re-id hypothesis
                 spatial_threshold=3.0,   # Maximum spatial distance for matching (meters)
                 temporal_threshold=2.0,  # Maximum temporal gap for matching (seconds)
                 velocity_threshold=1.0,  # Maximum velocity difference for matching
                 confidence_threshold=0.6):  # Minimum confidence for re-identification
        """
        Initialize the re-identification matcher.
        
        Args:
            max_occlusion_time: Maximum occlusion duration to track (seconds)
            spatial_threshold: Maximum spatial distance for matching (meters)
            temporal_threshold: Maximum temporal gap for matching (seconds)
            velocity_threshold: Maximum velocity difference for matching
            confidence_threshold: Minimum confidence score for re-identification
        """
        self.max_occlusion_time = max_occlusion_time
        self.spatial_threshold = spatial_threshold
        self.temporal_threshold = temporal_threshold
        self.velocity_threshold = velocity_threshold
        self.confidence_threshold = confidence_threshold
        
        # Track missing persons and their last known state
        self.missing_persons = {}  # person_id -> {disappearance_frame, last_position, last_velocity, last_covariance}
        
        # Track re-identification matches
        self.reidentification_matches = {}  # old_tracklet_id -> new_tracklet_id
        self.reidentification_scores = {}  # (old_id, new_id) -> confidence_score
    
    def register_missing_person(self, person_id, disappearance_frame, last_position, last_velocity, last_covariance):
        """
        Register a person as missing (disappeared from frame).
        
        Args:
            person_id: Unique identifier of the person
            disappearance_frame: Frame number when person disappeared
            last_position: Last known position [x, y]
            last_velocity: Last known velocity [v_x, v_y]
            last_covariance: Last known covariance matrix (2x2 for position)
        """
        self.missing_persons[person_id] = {
            'disappearance_frame': disappearance_frame,
            'last_position': last_position.copy(),
            'last_velocity': last_velocity.copy(),
            'last_covariance': last_covariance.copy()
        }
    
    def predict_position_at_frame(self, person_id, current_frame, dt=1/30.0):
        """
        Predict where a missing person should be at the current frame.
        
        Args:
            person_id: Person identifier
            current_frame: Current frame number
            dt: Time step (seconds)
            
        Returns:
            Predicted position [x, y]
        """
        if person_id not in self.missing_persons:
            return None
        
        person_data = self.missing_persons[person_id]
        disappearance_frame = person_data['disappearance_frame']
        last_position = person_data['last_position']
        last_velocity = person_data['last_velocity']
        
        # Calculate time elapsed since disappearance
        frames_elapsed = current_frame - disappearance_frame
        time_elapsed = frames_elapsed * dt
        
        # Predict position using constant velocity model
        predicted_position = last_position + last_velocity * time_elapsed
        
        return predicted_position
    
    def calculate_mahalanobis_distance(self, person_id, new_position, current_frame, dt=1/30.0):
        """
        Calculate Mahalanobis distance between predicted and new position.
        
        Args:
            person_id: Person identifier
            new_position: New tracklet position [x, y]
            current_frame: Current frame number
            dt: Time step (seconds)
            
        Returns:
            Mahalanobis distance (lower is better)
        """
        if person_id not in self.missing_persons:
            return np.inf
        
        person_data = self.missing_persons[person_id]
        predicted_position = self.predict_position_at_frame(person_id, current_frame, dt)
        
        if predicted_position is None:
            return np.inf
        
        # Get covariance and add process noise for the elapsed time
        covariance = person_data['last_covariance'].copy()
        
        # Add uncertainty due to time elapsed (process noise accumulation)
        frames_elapsed = current_frame - person_data['disappearance_frame']
        process_noise_std = 0.5  # Standard deviation of process noise
        covariance += np.eye(2) * (process_noise_std ** 2) * frames_elapsed
        
        # Ensure covariance is invertible
        try:
            cov_inv = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            return np.inf
        
        # Calculate Mahalanobis distance
        diff = new_position - predicted_position
        mahal_dist = np.sqrt(diff @ cov_inv @ diff.T)
        
        return mahal_dist
    
    def calculate_temporal_score(self, person_id, current_frame, dt=1/30.0):
        """
        Calculate temporal proximity score (higher is better).
        Persons that disappeared more recently get higher scores.
        
        Args:
            person_id: Person identifier
            current_frame: Current frame number
            dt: Time step (seconds)
            
        Returns:
            Temporal score (0 to 1)
        """
        if person_id not in self.missing_persons:
            return 0.0
        
        person_data = self.missing_persons[person_id]
        disappearance_frame = person_data['disappearance_frame']
        
        # Calculate time gap
        frames_gap = current_frame - disappearance_frame
        time_gap = frames_gap * dt
        
        # Score decreases with time gap
        # If time_gap > temporal_threshold, score is 0
        if time_gap > self.temporal_threshold:
            return 0.0
        
        # Linear decay: 1.0 at time_gap=0, 0.0 at time_gap=temporal_threshold
        temporal_score = 1.0 - (time_gap / self.temporal_threshold)
        return max(0.0, temporal_score)
    
    def calculate_spatial_score(self, person_id, new_position, current_frame, dt=1/30.0):
        """
        Calculate spatial proximity score (higher is better).
        Tracklets closer to predicted position get higher scores.
        
        Args:
            person_id: Person identifier
            new_position: New tracklet position [x, y]
            current_frame: Current frame number
            dt: Time step (seconds)
            
        Returns:
            Spatial score (0 to 1)
        """
        if person_id not in self.missing_persons:
            return 0.0
        
        predicted_position = self.predict_position_at_frame(person_id, current_frame, dt)
        
        if predicted_position is None:
            return 0.0
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(new_position - predicted_position)
        
        # Score decreases with distance
        # If distance > spatial_threshold, score is 0
        if distance > self.spatial_threshold:
            return 0.0
        
        # Linear decay: 1.0 at distance=0, 0.0 at distance=spatial_threshold
        spatial_score = 1.0 - (distance / self.spatial_threshold)
        return max(0.0, spatial_score)
    
    def calculate_velocity_consistency_score(self, person_id, new_velocity):
        """
        Calculate velocity consistency score (higher is better).
        Tracklets with similar velocity direction get higher scores.
        
        Args:
            person_id: Person identifier
            new_velocity: New tracklet velocity [v_x, v_y]
            
        Returns:
            Velocity consistency score (0 to 1)
        """
        if person_id not in self.missing_persons:
            return 0.5  # Neutral score if no previous velocity
        
        person_data = self.missing_persons[person_id]
        last_velocity = person_data['last_velocity']
        
        # Calculate velocity magnitude
        last_vel_mag = np.linalg.norm(last_velocity)
        new_vel_mag = np.linalg.norm(new_velocity)
        
        # If both velocities are near zero, they are consistent
        if last_vel_mag < 0.1 and new_vel_mag < 0.1:
            return 1.0
        
        # If one is near zero and the other is not, they are inconsistent
        if (last_vel_mag < 0.1) != (new_vel_mag < 0.1):
            return 0.2
        
        # Calculate angle between velocities
        cos_angle = np.dot(last_velocity, new_velocity) / (last_vel_mag * new_vel_mag + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        # Score based on angle: 1.0 for same direction, 0.0 for opposite
        angle_score = (cos_angle + 1.0) / 2.0
        
        return angle_score
    
    def calculate_confidence_score(self, person_id, new_position, new_velocity, current_frame, dt=1/30.0):
        """
        Calculate overall confidence score for re-identification.
        
        Args:
            person_id: Person identifier
            new_position: New tracklet position [x, y]
            new_velocity: New tracklet velocity [v_x, v_y]
            current_frame: Current frame number
            dt: Time step (seconds)
            
        Returns:
            Confidence score (0 to 1)
        """
        if person_id not in self.missing_persons:
            return 0.0
        
        # Calculate individual scores
        temporal_score = self.calculate_temporal_score(person_id, current_frame, dt)
        spatial_score = self.calculate_spatial_score(person_id, new_position, current_frame, dt)
        velocity_score = self.calculate_velocity_consistency_score(person_id, new_velocity)
        
        # Calculate Mahalanobis distance score (convert distance to score)
        mahal_dist = self.calculate_mahalanobis_distance(person_id, new_position, current_frame, dt)
        # Mahalanobis score: 1.0 if distance < 1.0, decreases as distance increases
        mahal_score = 1.0 / (1.0 + mahal_dist)
        
        # Weighted combination of scores
        weights = {
            'temporal': 0.2,
            'spatial': 0.3,
            'velocity': 0.2,
            'mahalanobis': 0.3
        }
        
        confidence = (
            weights['temporal'] * temporal_score +
            weights['spatial'] * spatial_score +
            weights['velocity'] * velocity_score +
            weights['mahalanobis'] * mahal_score
        )
        
        return confidence
    
    def match_new_tracklet(self, new_tracklet_id, new_position, new_velocity, current_frame, dt=1/30.0):
        """
        Attempt to match a new tracklet to a missing person.
        
        Args:
            new_tracklet_id: ID of the new tracklet
            new_position: Position of the new tracklet [x, y]
            new_velocity: Velocity of the new tracklet [v_x, v_y]
            current_frame: Current frame number
            dt: Time step (seconds)
            
        Returns:
            Tuple (matched_person_id, confidence_score) or (None, 0.0) if no match
        """
        best_match = None
        best_confidence = 0.0
        
        # Calculate confidence score for each missing person
        for person_id in list(self.missing_persons.keys()):
            confidence = self.calculate_confidence_score(
                person_id, new_position, new_velocity, current_frame, dt
            )
            
            # Store score for analysis
            self.reidentification_scores[(person_id, new_tracklet_id)] = confidence
            
            # Update best match
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = person_id
        
        # Only return match if confidence exceeds threshold
        if best_confidence >= self.confidence_threshold:
            self.reidentification_matches[best_match] = new_tracklet_id
            return best_match, best_confidence
        
        return None, 0.0
    
    def remove_expired_hypotheses(self, current_frame, dt=1/30.0):
        """
        Remove re-identification hypotheses that have expired (too old).
        
        Args:
            current_frame: Current frame number
            dt: Time step (seconds)
        """
        expired_persons = []
        
        for person_id, person_data in self.missing_persons.items():
            disappearance_frame = person_data['disappearance_frame']
            frames_elapsed = current_frame - disappearance_frame
            time_elapsed = frames_elapsed * dt
            
            # If occlusion duration exceeds max_occlusion_time, remove hypothesis
            if time_elapsed > self.max_occlusion_time:
                expired_persons.append(person_id)
        
        for person_id in expired_persons:
            del self.missing_persons[person_id]
            print(f"Removed expired re-identification hypothesis for person {person_id}")
    
    def get_reidentification_report(self):
        """
        Get a report of all re-identification matches and scores.
        
        Returns:
            Dictionary with re-identification information
        """
        return {
            'matches': self.reidentification_matches.copy(),
            'scores': self.reidentification_scores.copy(),
            'missing_persons': len(self.missing_persons),
            'total_matches': len(self.reidentification_matches)
        }


class HungarianReidentificationMatcher(ReidentificationMatcher):
    """
    Extended re-identification matcher using Hungarian algorithm for optimal matching
    when there are multiple new tracklets and multiple missing persons.
    """
    
    def match_multiple_tracklets(self, new_tracklets, current_frame, dt=1/30.0):
        """
        Match multiple new tracklets to missing persons using Hungarian algorithm.
        
        Args:
            new_tracklets: List of dicts with keys: 'id', 'position', 'velocity'
            current_frame: Current frame number
            dt: Time step (seconds)
            
        Returns:
            List of tuples (old_person_id, new_tracklet_id, confidence_score)
        """
        missing_person_ids = list(self.missing_persons.keys())
        
        # If no missing persons or no new tracklets, return empty
        if not missing_person_ids or not new_tracklets:
            return []
        
        # Build cost matrix (negative confidence scores)
        n_missing = len(missing_person_ids)
        n_new = len(new_tracklets)
        
        # Create cost matrix (larger dimension)
        cost_matrix = np.full((n_missing, n_new), -np.inf)
        
        # Fill cost matrix with negative confidence scores
        for i, person_id in enumerate(missing_person_ids):
            for j, tracklet in enumerate(new_tracklets):
                confidence = self.calculate_confidence_score(
                    person_id,
                    tracklet['position'],
                    tracklet['velocity'],
                    current_frame,
                    dt
                )
                cost_matrix[i, j] = -confidence  # Negative because we minimize cost
        
        # Solve assignment problem
        person_indices, tracklet_indices = linear_sum_assignment(cost_matrix)
        
        # Extract matches with confidence > threshold
        matches = []
        for person_idx, tracklet_idx in zip(person_indices, tracklet_indices):
            person_id = missing_person_ids[person_idx]
            tracklet_id = new_tracklets[tracklet_idx]['id']
            confidence = -cost_matrix[person_idx, tracklet_idx]
            
            if confidence >= self.confidence_threshold:
                self.reidentification_matches[person_id] = tracklet_id
                matches.append((person_id, tracklet_id, confidence))
        
        return matches
