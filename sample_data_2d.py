import numpy as np
from typing import List, Dict, Tuple
from uwb_ukf_2d import DT, run_ukf_simulation_2d

# --- Data Generation Parameters ---
CAMERA_FPS = 30
UWB_FPS = 3
TOTAL_TIME = 10 # seconds
TOTAL_FRAMES = int(TOTAL_TIME * CAMERA_FPS) # 300 frames

# --- 1. Generate True Trajectory (Global Cartesian) ---

def generate_true_trajectory_2d(total_frames: int) -> np.ndarray:
    """
    Generates a simple true 2D trajectory (x, y) for a person walking.
    Path: Starts at (1, 1), moves to (5, 5), then back to (1, 1).
    """
    t = np.arange(total_frames)
    
    # Phase 1: (1, 1) to (5, 5) - Frames 0 to 149
    x1 = np.linspace(1.0, 5.0, 150)
    y1 = np.linspace(1.0, 5.0, 150)
    
    # Phase 2: (5, 5) to (1, 1) - Frames 150 to 299
    x2 = np.linspace(5.0, 1.0, 150)
    y2 = np.linspace(5.0, 1.0, 150)
    
    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))
    
    trajectory = np.stack([x, y], axis=1)
    return trajectory

# --- 2. Generate Noisy UWB Measurements (Input to UKF) ---

def generate_uwb_measurements_2d(true_trajectory: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Generates noisy UWB measurements at 3 FPS (every 10th frame).
    
    :param true_trajectory: (TOTAL_FRAMES, 2) array of true (x, y) positions.
    :return: Dictionary mapping frame number (int) to (x, y) measurement (np.ndarray).
    """
    measurements = {}
    
    # UWB Measurement Noise (R from uwb_ukf_2d.py)
    R_pos_std = 0.1 # meters
    R_noise_cov = np.diag([R_pos_std**2, R_pos_std**2])
    
    # UWB measurements occur every CAMERA_FPS / UWB_FPS = 30 / 3 = 10 frames
    frame_interval = CAMERA_FPS // UWB_FPS
    
    for frame in range(0, TOTAL_FRAMES, frame_interval):
        true_pos = true_trajectory[frame]
        
        # Add noise
        noise = np.random.multivariate_normal([0, 0], R_noise_cov)
        noisy_pos = true_pos + noise
        
        measurements[frame] = noisy_pos
        
    return measurements

# --- 3. Generate Camera Tracklets (Output of MOT) ---

def generate_camera_tracklets_2d(true_trajectory: np.ndarray) -> List[Dict]:
    """
    Generates three camera tracklets:
    1. Opt-in person (matches UWB, full path)
    2. Non-opt-in person (different trajectory)
    3. Another non-opt-in person (static)
    """
    tracklets = []
    
    # --- Tracklet 1: Opt-in person (Full path) ---
    # Frames 0-299
    traj_1 = true_trajectory
    # Add small noise to camera tracklet (e.g., 5cm std dev)
    noise_1 = np.random.normal(0, 0.05, traj_1.shape)
    traj_1_noisy = traj_1 + noise_1
    
    tracklets.append({
        'id': 1,
        'trajectory': traj_1_noisy,
        'timestamps': list(range(TOTAL_FRAMES))
    })
    
    # --- Tracklet 2: Non-opt-in person (Different path) ---
    # Frames 0-299
    x_2 = np.linspace(6.0, 2.0, TOTAL_FRAMES)
    y_2 = np.linspace(2.0, 6.0, TOTAL_FRAMES)
    traj_2 = np.stack([x_2, y_2], axis=1)
    noise_2 = np.random.normal(0, 0.05, traj_2.shape)
    traj_2_noisy = traj_2 + noise_2
    
    tracklets.append({
        'id': 2,
        'trajectory': traj_2_noisy,
        'timestamps': list(range(TOTAL_FRAMES))
    })
    
    # --- Tracklet 3: Static person (Overlaps in time with all) ---
    # Frames 0-299
    static_pos = np.array([8.0, 8.0])
    traj_3 = np.tile(static_pos, (TOTAL_FRAMES, 1))
    noise_3 = np.random.normal(0, 0.01, traj_3.shape) # Very small noise for static
    traj_3_noisy = traj_3 + noise_3
    
    tracklets.append({
        'id': 3,
        'trajectory': traj_3_noisy,
        'timestamps': list(range(TOTAL_FRAMES))
    })
    
    return tracklets

# --- Main Data Generation Function ---

def generate_sample_data_2d() -> Tuple[List[Dict], List[Dict]]:
    """
    Generates all necessary sample data: UWB measurements and Camera tracklets.
    """
    np.random.seed(42) # for reproducibility
    
    # 1. True Trajectory
    true_traj = generate_true_trajectory_2d(TOTAL_FRAMES)
    
    # 2. UWB Measurements (Input to UKF)
    uwb_measurements = generate_uwb_measurements_2d(true_traj)
    
    # 3. Run UKF to get UWB Trajectory (Output of UKF)
    estimated_data = run_ukf_simulation_2d(uwb_measurements, TOTAL_FRAMES)
    
    # Extract position (x, y) and covariance P for all frames
    uwb_pos_traj = np.array([data['state'][[0, 2]] for data in estimated_data.values()])
    uwb_cov_traj = np.array([data['covariance'] for data in estimated_data.values()])
    
    uwb_trajectories = []
    uwb_trajectories.append({
        'id': 101, # UWB Tag ID
        'trajectory': uwb_pos_traj,
        'covariance': uwb_cov_traj,
        'timestamps': list(range(TOTAL_FRAMES))
    })
    
    # 4. Camera Tracklets
    cam_tracklets = generate_camera_tracklets_2d(true_traj)
    
    return uwb_trajectories, cam_tracklets

if __name__ == '__main__':
    uwb_trajectories, cam_tracklets = generate_sample_data_2d()
    
    print(f"Generated {len(uwb_trajectories)} UWB trajectory(ies).")
    print(f"UWB Trajectory 1 shape: {uwb_trajectories[0]['trajectory'].shape}")
    print(f"UWB Covariance 1 shape: {uwb_trajectories[0]['covariance'].shape}")
    
    print(f"\nGenerated {len(cam_tracklets)} Camera tracklet(s).")
    for tracklet in cam_tracklets:
        print(f"Tracklet ID {tracklet['id']}: Frames {tracklet['timestamps'][0]}-{tracklet['timestamps'][-1]}, Shape {tracklet['trajectory'].shape}")
