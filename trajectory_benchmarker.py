import numpy as np
from scipy.spatial.distance import cdist
from typing import List, Tuple, Dict

# --- Helper Functions for Trajectory Metrics ---

def mahalanobis_distance(p_uwb: np.ndarray, p_cam: np.ndarray, cov_inv: np.ndarray) -> float:
    """
    Calculates the Mahalanobis distance between a camera position and a UWB-UKF state.
    
    Args:
        p_uwb: UWB-UKF position (2D array: [x, y]).
        p_cam: Camera tracklet position (2D array: [x, y]).
        cov_inv: Inverse of the UWB-UKF covariance matrix (2x2 array).
        
    Returns:
        The Mahalanobis distance.
    """
    delta = p_cam - p_uwb
    # D = sqrt(delta.T @ cov_inv @ delta)
    return np.sqrt(delta.T @ cov_inv @ delta)

def euclidean_distance(p_uwb: np.ndarray, p_cam: np.ndarray) -> float:
    """
    Calculates the Euclidean distance between two points.
    """
    return np.linalg.norm(p_cam - p_uwb)

def frechet_distance(traj_a: np.ndarray, traj_b: np.ndarray) -> float:
    """
    Calculates the discrete Fréchet distance between two trajectories.
    
    NOTE: This is a simplified, non-optimized implementation for demonstration.
    For real-time use, optimized libraries or approximations are necessary.
    """
    n = traj_a.shape[0]
    m = traj_b.shape[0]
    
    if n == 0 or m == 0:
        return 0.0
    
    # Distance matrix (Euclidean distance between all pairs of points)
    D = cdist(traj_a, traj_b)
    
    # Initialize coupling matrix C
    C = np.full((n, m), -1.0)
    
    def calculate_coupling(i, j):
        if C[i, j] > -1:
            return C[i, j]
        
        if i == 0 and j == 0:
            C[i, j] = D[i, j]
        elif i > 0 and j == 0:
            C[i, j] = max(calculate_coupling(i - 1, 0), D[i, j])
        elif i == 0 and j > 0:
            C[i, j] = max(calculate_coupling(0, j - 1), D[i, j])
        elif i > 0 and j > 0:
            C[i, j] = max(D[i, j], min(
                calculate_coupling(i - 1, j),
                calculate_coupling(i - 1, j - 1),
                calculate_coupling(i, j - 1)
            ))
        else:
            C[i, j] = np.inf
            
        return C[i, j]

    return calculate_coupling(n - 1, m - 1)

def hausdorff_distance(traj_a: np.ndarray, traj_b: np.ndarray) -> float:
    """
    Calculates the Hausdorff distance between two trajectories.
    H(A, B) = max(h(A, B), h(B, A))
    h(A, B) = max_{a in A} min_{b in B} ||a - b||
    """
    # Calculate h(A, B)
    D_ab = cdist(traj_a, traj_b)
    h_ab = np.max(np.min(D_ab, axis=1))
    
    # Calculate h(B, A)
    D_ba = cdist(traj_b, traj_a)
    h_ba = np.max(np.min(D_ba, axis=1))
    
    return max(h_ab, h_ba)

# --- Real-Time Benchmarking Class ---

class RealTimeBenchmarker:
    def __init__(self, window_size_frames: int):
        self.window_size_frames = window_size_frames
        self.uwb_traj_buffer: List[np.ndarray] = []
        self.cam_traj_buffer: List[np.ndarray] = []
        self.cov_inv_buffer: List[np.ndarray] = []
        self.frame_count = 0

    def update(self, uwb_pos: np.ndarray, cam_pos: np.ndarray, cov_inv: np.ndarray):
        """
        Updates the buffers with new data and maintains the sliding window.
        """
        self.uwb_traj_buffer.append(uwb_pos)
        self.cam_traj_buffer.append(cam_pos)
        self.cov_inv_buffer.append(cov_inv)
        self.frame_count += 1

        # Maintain sliding window size
        if len(self.uwb_traj_buffer) > self.window_size_frames:
            self.uwb_traj_buffer.pop(0)
            self.cam_traj_buffer.pop(0)
            self.cov_inv_buffer.pop(0)

    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculates all trajectory metrics for the current sliding window.
        """
        if len(self.uwb_traj_buffer) < self.window_size_frames:
            return {"status": 0.0} # Not enough data for full window

        uwb_traj = np.array(self.uwb_traj_buffer)
        cam_traj = np.array(self.cam_traj_buffer)
        
        # 1. Mahalanobis Distance (Time-Averaged)
        mahalanobis_sum = 0.0
        for p_uwb, p_cam, cov_inv in zip(self.uwb_traj_buffer, self.cam_traj_buffer, self.cov_inv_buffer):
            mahalanobis_sum += mahalanobis_distance(p_uwb, p_cam, cov_inv)
        avg_mahalanobis = mahalanobis_sum / self.window_size_frames
        
        # 2. Euclidean Distance (Time-Averaged)
        euclidean_sum = 0.0
        for p_uwb, p_cam in zip(self.uwb_traj_buffer, self.cam_traj_buffer):
            euclidean_sum += euclidean_distance(p_uwb, p_cam)
        avg_euclidean = euclidean_sum / self.window_size_frames

        # 3. Fréchet Distance (Requires full trajectory)
        # NOTE: This is computationally expensive and should be run less frequently in a real system
        frechet_d = frechet_distance(uwb_traj, cam_traj)

        # 4. Hausdorff Distance (Requires full trajectory)
        hausdorff_d = hausdorff_distance(uwb_traj, cam_traj)

        return {
            "avg_mahalanobis": avg_mahalanobis,
            "avg_euclidean": avg_euclidean,
            "frechet_distance": frechet_d,
            "hausdorff_distance": hausdorff_d,
            "status": 1.0
        }

# --- Example Usage (Simulated Real-Time Loop) ---

if __name__ == "__main__":
    # Simulation Parameters
    FPS = 30
    WINDOW_SIZE_SEC = 1.0
    WINDOW_SIZE_FRAMES = int(WINDOW_SIZE_SEC * FPS)
    TOTAL_FRAMES = 100

    # Initialize Benchmarker
    benchmarker = RealTimeBenchmarker(WINDOW_SIZE_FRAMES)
    
    print(f"--- Real-Time Trajectory Benchmarking Simulation ---")
    print(f"Window Size: {WINDOW_SIZE_SEC}s ({WINDOW_SIZE_FRAMES} frames)")
    
    # Simulate data stream
    for frame in range(TOTAL_FRAMES):
        # 1. Simulate UWB-UKF Output (Position and Inverse Covariance)
        # UWB-UKF is tracking a person moving in a circle
        t = frame / FPS
        uwb_x = 5 * np.cos(t) + np.random.normal(0, 0.05)
        uwb_y = 5 * np.sin(t) + np.random.normal(0, 0.05)
        uwb_pos = np.array([uwb_x, uwb_y])
        
        # Simulate a stable, low-uncertainty inverse covariance matrix
        # (Covariance is low because the UKF is stable)
        cov_inv = np.linalg.inv(np.diag([0.1, 0.1])) 
        
        # 2. Simulate Camera Tracklet Position (Slightly offset and noisier)
        cam_x = 5 * np.cos(t + 0.1) + np.random.normal(0, 0.1)
        cam_y = 5 * np.sin(t + 0.1) + np.random.normal(0, 0.1)
        cam_pos = np.array([cam_x, cam_y])
        
        # Update benchmarker buffers
        benchmarker.update(uwb_pos, cam_pos, cov_inv)
        
        # Calculate metrics only after the window is full
        if frame >= WINDOW_SIZE_FRAMES - 1:
            metrics = benchmarker.calculate_metrics()
            
            if frame % 10 == 0: # Print every 10 frames for readability
                print(f"\nFrame {frame:03d} (t={t:.2f}s):")
                print(f"  Mahalanobis (Avg): {metrics['avg_mahalanobis']:.4f}")
                print(f"  Euclidean (Avg):   {metrics['avg_euclidean']:.4f}")
                print(f"  Fréchet Distance:  {metrics['frechet_distance']:.4f}")
                print(f"  Hausdorff Distance: {metrics['hausdorff_distance']:.4f}")
                
    print("\n--- Simulation Complete ---")
