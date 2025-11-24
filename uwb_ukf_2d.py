import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from scipy.linalg import block_diag

# --- Constants and Parameters ---
# Camera frame rate: 30 FPS -> DT = 1/30 s
DT = 1.0 / 30.0 

# --- UKF Functions ---

def f_x_2d(state, dt):
    """
    State transition function (motion model) for 2D.
    State: s = (x, vx, y, vy)
    Motion: Constant Velocity for x, y.
    """
    x, vx, y, vy = state
    
    # State transition matrix F for Constant Velocity (x, vx) and (y, vy)
    # F = [[1, dt, 0, 0],
    #      [0, 1, 0, 0],
    #      [0, 0, 1, dt],
    #      [0, 0, 0, 1]]
    
    # New state: x' = x + vx*dt, vx' = vx, y' = y + vy*dt, vy' = vy
    return np.array([x + vx * dt, vx, y + vy * dt, vy])

def h_x_2d(state):
    """
    Observation function (measurement model) for 2D.
    Transforms the state (x, vx, y, vy) to the UWB measurement (x, y).
    Since the UWB data is already in the world frame (x, y), this is a simple projection.
    """
    x, _, y, _ = state 
    return np.array([x, y])

def create_uwb_ukf_2d():
    """
    Initializes and returns the 2D UKF object.
    """
    # State dimension: 4 (x, vx, y, vy)
    dim_x = 4
    # Measurement dimension: 2 (x, y)
    dim_z = 2
    
    # Sigma points
    points = MerweScaledSigmaPoints(n=dim_x, alpha=.1, beta=2., kappa=-1)
    
    # Initialize UKF
    ukf = UKF(dim_x=dim_x, dim_z=dim_z, fx=f_x_2d, hx=h_x_2d, points=points, dt=DT)
    
    # Initial State (x, vx, y, vy) - Assumed starting point
    ukf.x = np.array([0., 0., 0., 0.])
    
    # Initial Covariance Matrix P
    ukf.P = np.diag([1.0, 1.0, 1.0, 1.0]) 
    
    # Process Noise Covariance Matrix Q
    # Q models the uncertainty in the motion model (Constant Velocity)
    # Q = block_diag(Q_cv, Q_cv)
    # Q_cv for (pos, vel)
    q_cv = np.array([[0.5*DT**2, DT], 
                     [DT, 1.0]]) * 0.1 # Tune this noise factor
    
    ukf.Q = block_diag(q_cv * 0.1, q_cv * 0.1)
    
    # Measurement Noise Covariance Matrix R
    # R models the uncertainty in the UWB measurements (x, y)
    # Example: 10cm standard deviation in x and y
    R_pos_std = 0.1 # meters
    
    ukf.R = np.diag([R_pos_std**2, R_pos_std**2])
    
    return ukf

# --- Example Usage (To be used with sample data in a later phase) ---
def run_ukf_simulation_2d(measurements: dict, total_frames: int):
    """
    Runs the 2D UKF with a sequence of UWB measurements.
    
    :param measurements: A dictionary mapping frame number (int) to (x, y) measurement (np.ndarray).
    :return: A dictionary mapping frame number (int) to (state, covariance)
    """
    ukf = create_uwb_ukf_2d()
    
    # Store the estimated states and covariances
    estimated_data = {}
    
    # The simulation runs for the duration of the camera recording (total_frames).
    for frame in range(total_frames):
        # Predict step (always runs at 30 FPS)
        ukf.predict()
        
        # Update step (only runs when a UWB measurement is available at 3 FPS)
        if frame in measurements:
            z = measurements[frame]
            ukf.update(z)
        
        # Store the estimated state (x, vx, y, vy) and covariance P
        estimated_data[frame] = {
            'state': ukf.x.copy(),
            'covariance': ukf.P.copy()
        }
        
    return estimated_data

if __name__ == '__main__':
    # Simple test case: stationary tag at (1.0, 1.0)
    print("Running 2D UKF test with stationary tag...")
    
    # Simulate 3 FPS measurements for 10 seconds (300 frames at 30 FPS)
    # Measurements at frames 0, 10, 20, ..., 290 (30 measurements)
    true_pos = np.array([1.0, 1.0])
    measurements = {}
    
    np.random.seed(42)
    R_pos_std = 0.1
    
    for i in range(30):
        frame = i * 10
        noise = np.random.normal(0, R_pos_std, 2)
        measurements[frame] = true_pos + noise
        
    estimated_data = run_ukf_simulation_2d(measurements)
    
    # Check convergence
    final_state = estimated_data[total_frames - 1]['state']
    final_pos = final_state[[0, 2]]
    
    print(f"Final state estimate: {final_state}")
    print(f"Final position (x, y): {final_pos}")
    
    # Expected final state: close to (1.0, 0.0, 1.0, 0.0)
    expected_pos = np.array([1.0, 1.0])
    error = np.linalg.norm(final_pos - expected_pos)
    print(f"Position error: {error:.4f} m")
    
    if error < 0.05:
        print("UKF converged successfully.")
    else:
        print("UKF convergence check failed (may need more tuning).")
