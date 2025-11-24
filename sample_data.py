import numpy as np
from typing import List, Dict
from uwb_ukf import ANCHOR_POS, TAG_HEIGHT, DT, create_uwb_ukf, run_ukf_simulation, h_x

# --- Data Generation Parameters ---
TOTAL_FRAMES = 100
FPS = 1.0 / DT # 10 FPS

# --- 1. Generate True Trajectory (Global Cartesian) ---

def generate_true_trajectory(total_frames: int) -> np.ndarray:
    """
    Generates a simple true trajectory (x, y, z) for a person walking.
    Path: Starts at (1, 1), moves to (5, 5), then back to (1, 1).
    """
    t = np.arange(total_frames)
    
    # Phase 1: (1, 1) to (5, 5) - Frames 0 to 49
    x1 = np.linspace(1.0, 5.0, 50)
    y1 = np.linspace(1.0, 5.0, 50)
    
    # Phase 2: (5, 5) to (1, 1) - Frames 50 to 99
    x2 = np.linspace(5.0, 1.0, 50)
    y2 = np.linspace(5.0, 1.0, 50)
    
    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))
    z = np.full(total_frames, TAG_HEIGHT)
    
    trajectory = np.stack([x, y, z], axis=1)
    return trajectory

# --- 2. Generate Noisy UWB Measurements (Input to UKF) ---

def generate_uwb_measurements(true_trajectory: np.ndarray) -> List[np.ndarray]:
    """
    Converts true trajectory to noisy polar UWB measurements.
    """
    measurements = []
    # Use the observation function from uwb_ukf.py (h_x)
    # The state for h_x is (x, vx, y, vy, z). We need to pad the true position.
    
    # UWB Measurement Noise (R from uwb_ukf.py)
    R_radial_std = 0.1 # meters
    R_angle_std = np.deg2rad(1.0) # radians
    R_noise_cov = np.diag([R_radial_std**2, R_angle_std**2, R_angle_std**2])
    
    for pos in true_trajectory:
        # Create a dummy state (x, 0, y, 0, z) for h_x
        dummy_state = np.array([pos[0], 0.0, pos[1], 0.0, pos[2]])
        
        # True polar measurement
        true_polar = h_x(dummy_state)
        
        # Add noise
        noise = np.random.multivariate_normal([0, 0, 0], R_noise_cov)
        noisy_polar = true_polar + noise
        
        measurements.append(noisy_polar)
        
    return measurements

# --- 3. Generate Camera Tracklets (Output of MOT) ---

def generate_camera_tracklets(true_trajectory: np.ndarray) -> List[Dict]:
    """
    Generates three camera tracklets:
    1. Opt-in person (matches UWB, split in the middle)
    2. Non-opt-in person (different trajectory)
    3. Another non-opt-in person (static)
    """
    tracklets = []
    
    # --- Tracklet 1a: Opt-in person (First half) ---
    # Frames 0-59 (60 frames)
    start_frame_1a, end_frame_1a = 0, 60
    traj_1a = true_trajectory[start_frame_1a:end_frame_1a]
    # Add small noise to camera tracklet
    noise_1a = np.random.normal(0, 0.05, traj_1a.shape)
    traj_1a_noisy = traj_1a + noise_1a
    
    tracklets.append({
        'id': 1,
        'trajectory': traj_1a_noisy,
        'timestamps': list(range(start_frame_1a, end_frame_1a))
    })
    
    # --- Tracklet 1b: Opt-in person (Second half, after occlusion) ---
    # Frames 65-99 (35 frames)
    start_frame_1b, end_frame_1b = 65, 100
    traj_1b = true_trajectory[start_frame_1b:end_frame_1b]
    noise_1b = np.random.normal(0, 0.05, traj_1b.shape)
    traj_1b_noisy = traj_1b + noise_1b
    
    tracklets.append({
        'id': 2,
        'trajectory': traj_1b_noisy,
        'timestamps': list(range(start_frame_1b, end_frame_1b))
    })
    
    # --- Tracklet 2: Non-opt-in person (Different path) ---
    # Frames 0-100
    x_2 = np.linspace(6.0, 2.0, TOTAL_FRAMES)
    y_2 = np.linspace(2.0, 6.0, TOTAL_FRAMES)
    z_2 = np.full(TOTAL_FRAMES, TAG_HEIGHT)
    traj_2 = np.stack([x_2, y_2, z_2], axis=1)
    noise_2 = np.random.normal(0, 0.05, traj_2.shape)
    traj_2_noisy = traj_2 + noise_2
    
    tracklets.append({
        'id': 3,
        'trajectory': traj_2_noisy,
        'timestamps': list(range(TOTAL_FRAMES))
    })
    
    # --- Tracklet 3: Static person (Overlaps in time with all) ---
    # Frames 0-100
    static_pos = np.array([8.0, 8.0, TAG_HEIGHT])
    traj_3 = np.tile(static_pos, (TOTAL_FRAMES, 1))
    noise_3 = np.random.normal(0, 0.01, traj_3.shape) # Very small noise for static
    traj_3_noisy = traj_3 + noise_3
    
    tracklets.append({
        'id': 4,
        'trajectory': traj_3_noisy,
        'timestamps': list(range(TOTAL_FRAMES))
    })
    
    return tracklets

# --- Main Data Generation Function ---

def generate_sample_data():
    """
    Generates all necessary sample data: UWB measurements and Camera tracklets.
    """
    np.random.seed(42) # for reproducibility
    
    # 1. True Trajectory
    true_traj = generate_true_trajectory(TOTAL_FRAMES)
    
    # 2. UWB Measurements (Input to UKF)
    uwb_measurements = generate_uwb_measurements(true_traj)
    
    # 3. Run UKF to get UWB Trajectory (Output of UKF)
    # The UKF also provides the covariance matrix P at each step.
    ukf = create_uwb_ukf()
    uwb_trajectories = []
    
    # We will only simulate one UWB tag for simplicity
    estimated_states = []
    estimated_covariances = []
    
    for z in uwb_measurements:
        ukf.predict()
        ukf.update(z)
        estimated_states.append(ukf.x.copy())
        estimated_covariances.append(ukf.P.copy())
        
    estimated_states = np.array(estimated_states)
    estimated_covariances = np.array(estimated_covariances)
    
    # Extract position (x, y, z) from state (x, vx, y, vy, z)
    uwb_pos_traj = estimated_states[:, [0, 2, 4]]
    
    uwb_trajectories.append({
        'id': 101, # UWB Tag ID
        'trajectory': uwb_pos_traj,
        'covariance': estimated_covariances,
        'timestamps': list(range(TOTAL_FRAMES))
    })
    
    # 4. Camera Tracklets
    cam_tracklets = generate_camera_tracklets(true_traj)
    
    return uwb_trajectories, cam_tracklets

if __name__ == '__main__':
    uwb_trajectories, cam_tracklets = generate_sample_data()
    
    print(f"Generated {len(uwb_trajectories)} UWB trajectory(ies).")
    print(f"UWB Trajectory 1 shape: {uwb_trajectories[0]['trajectory'].shape}")
    print(f"UWB Covariance 1 shape: {uwb_trajectories[0]['covariance'].shape}")
    
    print(f"\nGenerated {len(cam_tracklets)} Camera tracklet(s).")
    for tracklet in cam_tracklets:
        print(f"Tracklet ID {tracklet['id']}: Frames {tracklet['timestamps'][0]}-{tracklet['timestamps'][-1]}, Shape {tracklet['trajectory'].shape}")
