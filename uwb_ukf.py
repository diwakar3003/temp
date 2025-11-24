import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from scipy.linalg import block_diag

# --- Constants and Parameters (Based on Paper's description and typical UWB setup) ---
# The paper states the state vector is s = (x, vx, y, vy, z)
# x, y, z are position, vx, vy are on-ground velocities.
# The motion model is Constant Velocity for x and y, and Constant Position for z.
# The observation is in polar coordinates: z_t = (z_radial, z_azimuth, z_elevation)

# Time step (e.g., 10 FPS -> dt = 0.1s)
DT = 0.1

# Anchor position and orientation (Calibration parameters - assumed for this example)
# In a real system, these would be determined by the calibration step (Sec. IV-A)
ANCHOR_POS = np.array([0.0, 0.0, 2.0]) # Example: Anchor at (0, 0, 2m)
ANCHOR_ORIENTATION = np.array([0.0, 0.0, 0.0]) # Example: No rotation (yaw, pitch, roll)

# Tag height (Calibration parameter - assumed for this example)
TAG_HEIGHT = 1.27 # Example value from the paper's Table 1

# --- UKF Functions ---

def f_x(state, dt):
    """
    State transition function (motion model).
    State: s = (x, vx, y, vy, z)
    Motion: Constant Velocity for x, y. Constant Position for z.
    """
    x, vx, y, vy, z = state
    
    # State transition matrix F for Constant Velocity (x, vx) and (y, vy)
    # and Constant Position (z)
    # F = [[1, dt, 0, 0, 0],
    #      [0, 1, 0, 0, 0],
    #      [0, 0, 1, dt, 0],
    #      [0, 0, 0, 1, 0],
    #      [0, 0, 0, 0, 1]]
    
    # New state: x' = x + vx*dt, vx' = vx, y' = y + vy*dt, vy' = vy, z' = z
    return np.array([x + vx * dt, vx, y + vy * dt, vy, z])

def h_x(state):
    """
    Observation function (measurement model).
    Transforms the state (x, vx, y, vy, z) in global Cartesian coordinates
    to the UWB anchor's polar coordinates (radial, azimuth, elevation).
    
    The paper mentions:
    1. Affine transformation into anchor's local Cartesian system.
    2. Conversion to polar coordinates.
    
    For simplicity in this example, we assume the global coordinate system
    is aligned with the anchor's system, and the anchor is at the origin (0,0,0)
    in its own frame. We will use the ANCHOR_POS to transform from global to anchor frame.
    """
    x, _, y, _, z = state # Only position components are needed for observation
    
    # 1. Transform from Global Cartesian to Anchor's Local Cartesian
    # Assuming ANCHOR_POS is in the Global Cartesian frame
    local_pos = np.array([x, y, z]) - ANCHOR_POS
    lx, ly, lz = local_pos
    
    # 2. Convert to Polar Coordinates (radial, azimuth, elevation)
    # Radial distance (range)
    radial = np.sqrt(lx**2 + ly**2 + lz**2)
    
    # Azimuth (angle in the xy-plane, from x-axis)
    azimuth = np.arctan2(ly, lx) # atan2(y, x)
    
    # Elevation (angle from the xy-plane)
    # Note: Some systems define elevation from the z-axis (polar angle), 
    # but the paper implies angle from the ground plane (xy-plane).
    # We use the standard definition: arcsin(z / radial)
    elevation = np.arcsin(lz / radial) if radial > 1e-6 else 0.0
    
    return np.array([radial, azimuth, elevation])

def create_uwb_ukf():
    """
    Initializes and returns the UKF object.
    """
    # State dimension: 5 (x, vx, y, vy, z)
    dim_x = 5
    # Measurement dimension: 3 (radial, azimuth, elevation)
    dim_z = 3
    
    # Sigma points - MerweScaledSigmaPoints is a good default choice
    points = MerweScaledSigmaPoints(n=dim_x, alpha=.1, beta=2., kappa=-1)
    
    # Initialize UKF
    ukf = UKF(dim_x=dim_x, dim_z=dim_z, fx=f_x, hx=h_x, points=points, dt=DT)
    
    # Initial State (x, vx, y, vy, z) - Assumed starting point
    ukf.x = np.array([0., 0., 0., 0., TAG_HEIGHT])
    
    # Initial Covariance Matrix P
    # High uncertainty initially
    ukf.P = np.diag([1.0, 1.0, 1.0, 1.0, 1.0]) 
    
    # Process Noise Covariance Matrix Q
    # Q models the uncertainty in the motion model (Constant Velocity)
    # Q = block_diag(Q_cv, Q_cv, Q_const_pos)
    # Q_cv for (pos, vel)
    q_cv = np.array([[0.5*DT**2, DT], 
                     [DT, 1.0]]) * 0.1 # Tune this noise factor
    
    # Q_const_pos for z
    q_const_pos = np.array([[0.01]]) # Very small noise for constant position
    
    ukf.Q = block_diag(q_cv * 0.1, q_cv * 0.1, q_const_pos)
    
    # Measurement Noise Covariance Matrix R
    # R models the uncertainty in the UWB measurements (radial, azimuth, elevation)
    # These values are highly dependent on the UWB hardware
    # Example: 10cm radial std dev, 1 degree azimuth/elevation std dev
    R_radial_std = 0.1 # meters
    R_angle_std = np.deg2rad(1.0) # radians
    
    ukf.R = np.diag([R_radial_std**2, R_angle_std**2, R_angle_std**2])
    
    return ukf

# --- Example Usage (To be used with sample data in a later phase) ---
def run_ukf_simulation(measurements):
    """
    Runs the UKF with a sequence of UWB measurements.
    
    :param measurements: A list of (radial, azimuth, elevation) tuples.
    :return: A list of estimated states (x, vx, y, vy, z).
    """
    ukf = create_uwb_ukf()
    
    # Store the estimated states
    estimated_states = []
    
    for z in measurements:
        # Predict step
        ukf.predict()
        
        # Update step
        ukf.update(z)
        
        # Store the estimated state (x, vx, y, vy, z)
        estimated_states.append(ukf.x.copy())
        
    return np.array(estimated_states)

if __name__ == '__main__':
    # Simple test case: stationary tag
    print("Running UKF test with stationary tag...")
    
    # True position: (1.0, 1.0, 1.27)
    # Anchor at (0.0, 0.0, 2.0)
    # Local position: (1.0, 1.0, -0.73)
    # Radial: sqrt(1^2 + 1^2 + (-0.73)^2) = 1.618
    # Azimuth: atan2(1, 1) = 0.785 rad (45 deg)
    # Elevation: asin(-0.73 / 1.618) = -0.47 rad (-27 deg)
    
    true_radial = 1.618
    true_azimuth = 0.785
    true_elevation = -0.47
    
    # Simulate noisy measurements
    np.random.seed(42)
    measurements = []
    for _ in range(50):
        noise = np.random.multivariate_normal([0, 0, 0], np.diag([0.1**2, np.deg2rad(1.0)**2, np.deg2rad(1.0)**2]))
        measurements.append(np.array([true_radial, true_azimuth, true_elevation]) + noise)
        
    estimated_states = run_ukf_simulation(measurements)
    
    print(f"Initial state estimate: {estimated_states[0]}")
    print(f"Final state estimate: {estimated_states[-1]}")
    
    # Expected final state: close to (1.0, 0.0, 1.0, 0.0, 1.27)
    # The UKF is initialized at (0, 0, 0, 0, 1.27), so it should converge to (1.0, 0.0, 1.0, 0.0, 1.27)
    
    # Extract position (x, y, z)
    final_pos = estimated_states[-1][[0, 2, 4]]
    print(f"Final position (x, y, z): {final_pos}")
    
    # Check convergence (simple check)
    expected_pos = np.array([1.0, 1.0, TAG_HEIGHT])
    error = np.linalg.norm(final_pos - expected_pos)
    print(f"Position error: {error:.4f} m")
    
    if error < 0.1:
        print("UKF converged successfully.")
    else:
        print("UKF convergence check failed (may need more tuning).")
