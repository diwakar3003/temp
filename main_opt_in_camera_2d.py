import numpy as np
from sample_data_2d import generate_sample_data_2d
from trajectory_matching_2d import calculate_cost_matrix_2d, solve_trajectory_matching_2d
from typing import List, Tuple

# --- Main Integration Script ---

def main():
    """
    Integrates the 2D UWB UKF tracking and the 2D Trajectory Matching logic
    using the generated sample data.
    """
    print("--- Opt-in Camera System Core Logic Simulation (2D World Frame) ---")
    
    # 1. Generate Sample Data
    print("\n1. Generating sample UWB and Camera tracklet data...")
    uwb_trajectories, cam_tracklets = generate_sample_data_2d()
    
    print(f"   - UWB Trajectories: {len(uwb_trajectories)}")
    print(f"   - Camera Tracklets: {len(cam_tracklets)}")
    
    # The sample data is structured as:
    # UWB Trajectory 1 (ID 101) -> True Opt-in Person
    # Camera Tracklet 1 (ID 1) -> Opt-in Person (Full path)
    # Camera Tracklet 2 (ID 2) -> Non-opt-in Person (Full path)
    # Camera Tracklet 3 (ID 3) -> Static Non-opt-in Person (Full time)
    
    # Expected Match: UWB 101 should match Cam 1.
    
    # 2. Calculate Cost Matrix (Similarity)
    # We use a cost threshold (c_th) of 1.0, meaning the average Mahalanobis distance
    # must be less than 1.0 for a potential match.
    C, time_overlap = calculate_cost_matrix_2d(
        uwb_trajectories=uwb_trajectories, 
        cam_tracklets=cam_tracklets, 
        c_th=10.0
    )
    
    print("\n2. Calculated Similarity Matrix (C_ij):")
    # Print the similarity matrix C (UWB tags x Camera tracklets)
    # Row indices are UWB tags, Column indices are Camera tracklets
    print(C)
    
    # 3. Solve Trajectory Matching (Linear Assignment Problem)
    assignments = solve_trajectory_matching_2d(C)
    
    print("\n3. Trajectory Matching Results (UWB Index -> Camera Tracklet Index):")
    
    # Map indices back to IDs for clear output
    results = []
    for uwb_idx, cam_idx in assignments:
        uwb_id = uwb_trajectories[uwb_idx]['id']
        cam_id = cam_tracklets[cam_idx]['id']
        similarity = C[uwb_idx, cam_idx]
        
        results.append({
            'UWB_ID': uwb_id,
            'Camera_Tracklet_ID': cam_id,
            'Similarity': similarity
        })
        
    # Sort by similarity (descending)
    results.sort(key=lambda x: x['Similarity'], reverse=True)
    
    # Print results
    for res in results:
        print(f"   - UWB Tag {res['UWB_ID']} matched to Camera Tracklet {res['Camera_Tracklet_ID']} (Similarity: {res['Similarity']:.4f})")
        
    # 4. Identify Opt-in Individuals
    opt_in_tracklet_ids = [res['Camera_Tracklet_ID'] for res in results]
    
    print("\n4. Identified Opt-in Camera Tracklets (for Masking):")
    print(f"   - Tracklet IDs: {opt_in_tracklet_ids}")
    
    # Check the similarity of all matches for UWB 101
    print("\n--- Similarity Check for All Potential Matches (UWB 101) ---")
    # UWB 101 is index 0
    # Cam 1 is index 0, Cam 2 is index 1, Cam 3 is index 2
    print(f"   - UWB 101 (idx 0) to Cam 1 (idx 0): Similarity = {C[0, 0]:.4f}")
    print(f"   - UWB 101 (idx 0) to Cam 2 (idx 1): Similarity = {C[0, 1]:.4f}")
    print(f"   - UWB 101 (idx 0) to Cam 3 (idx 2): Similarity = {C[0, 2]:.4f}")
    
if __name__ == '__main__':
    main()
