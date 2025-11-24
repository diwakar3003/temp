import numpy as np
from sample_data import generate_sample_data
from trajectory_matching import calculate_cost_matrix, solve_trajectory_matching
from typing import List, Tuple

# --- Main Integration Script ---

def main():
    """
    Integrates the UWB UKF tracking and the Trajectory Matching logic
    using the generated sample data.
    """
    print("--- Opt-in Camera System Core Logic Simulation ---")
    
    # 1. Generate Sample Data
    print("\n1. Generating sample UWB and Camera tracklet data...")
    uwb_trajectories, cam_tracklets = generate_sample_data()
    
    print(f"   - UWB Trajectories: {len(uwb_trajectories)}")
    print(f"   - Camera Tracklets: {len(cam_tracklets)}")
    
    # The sample data is structured as:
    # UWB Trajectory 1 (ID 101) -> True Opt-in Person
    # Camera Tracklet 1 (ID 1) -> Opt-in Person (First half)
    # Camera Tracklet 2 (ID 2) -> Opt-in Person (Second half)
    # Camera Tracklet 3 (ID 3) -> Non-opt-in Person (Full path)
    # Camera Tracklet 4 (ID 4) -> Static Non-opt-in Person (Full time)
    
    # Expected Match: UWB 101 should match Cam 1 and Cam 2.
    
    # 2. Calculate Cost Matrix (Similarity)
    # We use a cost threshold (c_th) of 1.0, meaning the average Mahalanobis distance
    # must be less than 1.0 for a potential match.
    C, time_overlap = calculate_cost_matrix(
        uwb_trajectories=uwb_trajectories, 
        cam_tracklets=cam_tracklets, 
        c_th=10.0
    )
    
    print("\n2. Calculated Similarity Matrix (C_ij):")
    # Print the similarity matrix C (UWB tags x Camera tracklets)
    # Row indices are UWB tags, Column indices are Camera tracklets
    print(C)
    
    # 3. Solve Trajectory Matching (Linear Assignment Problem)
    # This will find the best one-to-one match (UWB -> Camera Tracklet)
    # The paper implies a one-to-many match is possible (one UWB tag to multiple tracklets
    # of the same person due to occlusion), but the implemented solver is one-to-one.
    assignments = solve_trajectory_matching(C, time_overlap)
    
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
    # In a real system, the UWB tag ID is the identifier for the opt-in person.
    # The camera tracklets matched to this UWB ID are the ones to keep.
    
    opt_in_tracklet_ids = [res['Camera_Tracklet_ID'] for res in results]
    
    print("\n4. Identified Opt-in Camera Tracklets (for Masking):")
    print(f"   - Tracklet IDs: {opt_in_tracklet_ids}")
    
    # The expected result is that UWB 101 matches the tracklet with the highest similarity.
    # Since tracklets 1 and 2 belong to the same person as UWB 101, they should have
    # high similarity. The LAP will pick the one with the highest score.
    # If we had a one-to-many solver, both 1 and 2 would be matched to 101.
    
    # Let's check the similarity of the expected matches (UWB 101 is index 0)
    # Cam 1 is index 0, Cam 2 is index 1, Cam 3 is index 2, Cam 4 is index 3
    print("\n--- Similarity Check for Expected Matches ---")
    print(f"   - UWB 101 (idx 0) to Cam 1 (idx 0): Similarity = {C[0, 0]:.4f}")
    print(f"   - UWB 101 (idx 0) to Cam 2 (idx 1): Similarity = {C[0, 1]:.4f}")
    print(f"   - UWB 101 (idx 0) to Cam 3 (idx 2): Similarity = {C[0, 2]:.4f}")
    print(f"   - UWB 101 (idx 0) to Cam 4 (idx 3): Similarity = {C[0, 3]:.4f}")
    
    # The highest similarity should be for Cam 1 or Cam 2, which are the true matches.
    
if __name__ == '__main__':
    main()
