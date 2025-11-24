import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import mahalanobis
from typing import List, Dict, Tuple

# --- Trajectory Matching Logic (Simplified 2D) ---

def calculate_mahalanobis_distance_2d(point_uwb: np.ndarray, point_cam: np.ndarray, covariance_matrix: np.ndarray) -> float:
    """
    Calculates the Mahalanobis distance D(p_i^t, p_j^t) between a UWB position
    and a camera tracklet position, using the UKF's covariance matrix.
    
    The UKF state is (x, vx, y, vy). The position is (x, y).
    We use the sub-matrix of P corresponding to (x, y).
    
    :param point_uwb: UWB position (x, y)
    :param point_cam: Camera position (x, y)
    :param covariance_matrix: Full 4x4 UKF covariance matrix P
    :return: Mahalanobis distance
    """
    # Extract the covariance sub-matrix for position (x, y)
    # Indices for (x, y) in (x, vx, y, vy) are 0, 2
    indices = [0, 2]
    P_pos = covariance_matrix[np.ix_(indices, indices)]
    
    # Mahalanobis distance requires the inverse of the covariance matrix
    try:
        P_inv = np.linalg.inv(P_pos)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if matrix is singular
        P_inv = np.linalg.pinv(P_pos)
        
    # Mahalanobis distance: sqrt((u-v).T * inv(C) * (u-v))
    return mahalanobis(point_uwb, point_cam, P_inv)

def calculate_cost_matrix_2d(uwb_trajectories: List[Dict], cam_tracklets: List[Dict], c_th: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the cost matrix C_ij for the linear assignment problem (LAP) for 2D data.
    
    :param uwb_trajectories: List of UWB trajectories (Dict with 'trajectory' and 'covariance' keys)
    :param cam_tracklets: List of camera tracklets (Dict with 'trajectory' key)
    :param c_th: Cost threshold (average Mahalanobis distance)
    :return: Cost matrix C (Similarity) and a dummy time_overlap matrix
    """
    N_uwb = len(uwb_trajectories)
    N_cam = len(cam_tracklets)
    
    # Initialize cost matrix with a high value (representing no match or high cost)
    C = np.full((N_uwb, N_cam), -np.inf) # Use -inf for no overlap, as we maximize
    
    # Dummy time_overlap matrix (Constraint 4 is ignored for standard LAP)
    time_overlap = np.zeros((N_cam, N_cam), dtype=bool)
    
    # --- Calculate Cost (Similarity) Matrix C ---
    for i in range(N_uwb):
        uwb_data = uwb_trajectories[i]
        uwb_pos_traj = uwb_data['trajectory'] # (N_frames, 2) array of (x, y)
        uwb_cov_traj = uwb_data['covariance'] # (N_frames, 4, 4) array of P matrices
        uwb_timestamps = uwb_data['timestamps']
        
        for j in range(N_cam):
            cam_data = cam_tracklets[j]
            cam_pos_traj = cam_data['trajectory'] # (N_frames, 2) array of (x, y)
            cam_timestamps = cam_data['timestamps']
            
            # Find intersecting timestamps (T_i \cap T_j)
            common_timestamps = sorted(list(set(uwb_timestamps).intersection(set(cam_timestamps))))
            
            if not common_timestamps:
                continue
                
            total_distance = 0.0
            count = 0
            
            for t in common_timestamps:
                # The trajectories are indexed by frame number (timestamp)
                # Since the data is generated to be dense at the camera rate (30 FPS),
                # the timestamp is the index.
                
                # Get the position and covariance at frame t
                p_uwb = uwb_pos_traj[t]
                P_t = uwb_cov_traj[t]
                p_cam = cam_pos_traj[t]
                
                # Calculate Mahalanobis distance
                dist = calculate_mahalanobis_distance_2d(p_uwb, p_cam, P_t)
                total_distance += dist
                count += 1
                
            # Calculate the time-averaged distance (Eq. 6)
            avg_distance = total_distance / count
            
            # Similarity is the inverse of the average distance
            similarity = 1.0 / (avg_distance + 1e-6)
            C[i, j] = similarity
            
            # Apply constraint (5): x_ij = 0 if c_ij < c_th (i.e., avg_distance > c_th)
            if avg_distance > c_th:
                C[i, j] = -np.inf # Effectively sets x_ij = 0 in maximization
                
    return C, time_overlap

def solve_trajectory_matching_2d(C: np.ndarray) -> List[Tuple[int, int]]:
    """
    Solves the constrained linear optimization problem (Eq. 2-5) for 2D data.
    
    :param C: The similarity matrix (c_ij)
    :return: List of (uwb_index, cam_index) tuples representing the assignment
    """
    # Convert similarity (maximization) to cost (minimization)
    max_similarity = np.max(C[C != -np.inf]) if np.any(C != -np.inf) else 1.0
    
    # Cost matrix: Cost = max_similarity - Similarity
    cost_matrix = max_similarity - C
    cost_matrix[C == -np.inf] = np.inf
    
    # Solve the Linear Assignment Problem (LAP)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Filter out assignments where the original cost was -inf
    assignments = []
    for i, j in zip(row_ind, col_ind):
        if C[i, j] != -np.inf:
            assignments.append((i, j))
            
    return assignments

# --- Data Structure Definitions (for clarity) ---

# UWB Trajectory Data Structure:
# {
#     'id': int,
#     'trajectory': np.ndarray, # (N_frames, 2) array of (x, y) positions
#     'covariance': np.ndarray, # (N_frames, 4, 4) array of P matrices (x, vx, y, vy)
#     'timestamps': List[int] # List of frame numbers (0 to N_frames-1)
# }

# Camera Tracklet Data Structure:
# {
#     'id': int,
#     'trajectory': np.ndarray, # (N_frames, 2) array of (x, y) positions
#     'timestamps': List[int] # List of frame numbers (0 to N_frames-1)
# }
