# Real-Time Multi-Person Opt-in Camera System: Technical Documentation (V2)

**Author:** Manus AI
**Date:** December 09, 2025
**Version:** 2.0 (Expanded Detail)

## 1. Executive Summary and Core Innovation

This document details the architecture and implementation of a real-time, multi-person **Opt-in Camera System** optimized for low-latency inference on edge devices. The system's primary function is to reliably identify individuals who have explicitly consented to be tracked by carrying an Ultra-Wideband (UWB) tag.

The system's core innovation is the use of the **Unscented Kalman Filter (UKF)** as a temporal and probabilistic bridge to fuse sparse UWB location data (3 FPS) with high-rate camera tracklet data (30 FPS). This fusion creates a continuous, uncertainty-weighted trajectory for each opt-in person, enabling robust identification via **Mahalanobis distance-based Trajectory Matching**.

### 1.1. Importance of UWB for Opt-in Identification

The use of UWB technology is fundamental to the system's privacy-preserving design and its technical capability for identification.

| Aspect | UWB Role | Advantage |
| :--- | :--- | :--- |
| **Privacy & Consent** | **Explicit Opt-in:** The UWB tag acts as a physical, verifiable token of consent. Only persons carrying a tag are tracked. | Ensures the system adheres to privacy regulations by design. |
| **Identity Anchor** | **Unique ID:** Each UWB tag transmits a unique, persistent identifier. | Provides a ground-truth identity that is invariant to visual changes (e.g., clothing, occlusion). |
| **Location Accuracy** | **High Precision:** UWB provides centimeter-level localization accuracy in the world frame. | Essential for accurate probabilistic matching against visual data. |
| **Occlusion Robustness** | **Non-Line-of-Sight:** UWB signals penetrate walls and objects better than visual sensors. | The UKF can be updated by UWB even when the person is visually occluded, maintaining state continuity. |

Without the UWB data, the system would be a standard visual tracking system, unable to verify consent or maintain identity through long-term occlusions.

## 2. System Architecture and Data Flow

The system operates on a decoupled, multi-rate architecture to ensure real-time compliance, with the entire pipeline synchronized to the camera's frame rate (30 FPS).

| Module | Rate | Purpose |
| :--- | :--- | :--- |
| **Data Sources** | 30 FPS (Camera), 3 FPS (UWB) | Provide asynchronous, noisy input data. |
| **Multi-Person Tracker** | 30 FPS | Manages individual UKF states and runs prediction/update steps. |
| **Trajectory Matcher** | 10 FPS | Solves the optimal assignment problem between UWB tags and camera tracklets. |
| **Output** | 10 FPS | Final identified person (UWB ID $\rightarrow$ Tracklet ID). |

### 2.1. Detailed Flow Diagram Steps

The process is executed in a continuous loop, with each step designed for low-latency processing:

1.  **Data Ingestion (30 FPS):** The `vehicle_data_streamer.py` module yields the current frame's data, including a list of camera tracklets (position, ID) and any available UWB measurements (position, ID).
2.  **UKF Prediction (30 FPS):** The `MultiPersonTracker` iterates through all active UWB tags. For each tag, the `StreamingUWBUKF2D` instance calls `ukf.predict()`. This step uses the Constant Velocity motion model to project the state forward by $\Delta t = 1/30$ seconds.
3.  **UKF Update (3 FPS Trigger):** The `MultiPersonTracker` checks if a UWB measurement is available for a tag. If so, the `ukf.update()` step is executed, using the UWB position as the observation to correct the predicted state. If no UWB data is available, the filter relies solely on the prediction, and its uncertainty (covariance) grows.
4.  **State Buffering (30 FPS):** The current UKF state (position and covariance) and the camera tracklet data are stored in a circular buffer within the `RealTimeTrajectoryMatcher`.
5.  **Trajectory Matching Trigger (10 FPS):** Every third frame, the matching process is triggered.
6.  **Cost Matrix Calculation:** The matcher calculates the Mahalanobis distance cost matrix between all active UWB tags and all visible camera tracklets over a 1-second sliding window.
7.  **LAP Solution:** The Hungarian algorithm is applied to the cost matrix to find the optimal one-to-one assignment that minimizes the total cost.
8.  **Output:** The optimal assignment (UWB ID $\rightarrow$ Tracklet ID) is reported, providing the final, verified identity of the opt-in person.

## 3. Module Implementation Details

### 3.1. Module: `uwb_ukf_2d_streaming.py` (Streaming UKF)

This module implements the core state estimation logic, acting as the temporal bridge.

| Component | Detail |
| :--- | :--- |
| **State Vector ($x$)** | 4D: $(x, v_x, y, v_y)$. Position and velocity in the world frame. |
| **Motion Model** | **Constant Velocity (CV):** Assumes constant velocity between frames. The state transition matrix $\mathbf{F}$ is applied to propagate the state. |
| **Observation Model ($h$)** | **Direct Position:** $h(x) = [x, y]^T$. The UKF is driven by the UWB measurement, which is a direct observation of the person's position. |
| **Prediction Rate** | **30 FPS:** Ensures a continuous, high-rate trajectory estimate, bridging the 3 FPS UWB gap. |
| **Update Trigger** | **3 FPS:** The filter's state is corrected only when a new UWB measurement arrives. |
| **Key Feature** | The UKF's output includes the **Covariance Matrix ($\Sigma$)**, which quantifies the uncertainty of the position estimate. This is crucial for the Mahalanobis distance. |

### 3.2. Module: `multi_person_tracker.py` (Tracker Manager)

This module is the central orchestrator, managing the lifecycle and state of multiple persons.

| Method | Purpose |
| :--- | :--- |
| `predict_all()` | Runs the prediction step for all active UKF instances at 30 FPS. |
| `update_all_uwb()` | Checks for new UWB data. If a tag is new, it initializes a new UKF. If a tag is existing, it calls `ukf.update()`. |
| **Occlusion Handling** | When a camera tracklet is lost, the UKF continues to predict based on the motion model and UWB updates. The identity is maintained, and the growing covariance reflects the increased uncertainty. |
| **Late Arrival** | A new UWB tag is detected and a new UKF instance is initialized, allowing the person to be tracked from the moment they opt-in. |

## 3.3. Module: `trajectory_matching_2d_streaming.py` (Real-Time Matcher)

This module performs the probabilistic assignment of camera tracklets to UWB tags.

| Component | Detail |
| :--- | :--- |
| **Matching Rate** | **10 FPS:** Decoupled from the 30 FPS tracking rate to reduce computational load and increase temporal stability. |
| **Sliding Window** | **1-second window (30 frames):** The cost is calculated over the most recent 1 second of data, ensuring the matching is based on current movement patterns. |
| **Cost Matrix** | $C_{ij}$ is the negative time-averaged Mahalanobis similarity between UWB tag $i$ and camera tracklet $j$. |
| **LAP Feasibility Fix** | Costs exceeding the `cost_threshold` (e.g., 5.0) are replaced with a large, finite number (`MAX_COST = 1000.0`) to prevent the "infeasible cost matrix" error from the LAP solver. |

## 4. Trajectory Matching Methodologies: A Detailed Comparison

The choice of similarity metric is the most critical factor in the identification process. The system uses **Mahalanobis Distance** for its probabilistic advantages over simpler metrics.

### 4.1. Mahalanobis Distance (System Choice)

The Mahalanobis distance ($D$) measures the distance between a point and a distribution, accounting for the covariance ($\Sigma$) of the distribution.

$$
D = \sqrt{(p_{cam} - p_{uwb})^T \Sigma^{-1} (p_{cam} - p_{uwb})}
$$

| Advantage | Disadvantage |
| :--- | :--- |
| **Probabilistic:** Accounts for the UKF's uncertainty. A match is penalized less if the UKF is highly uncertain (high $\Sigma$). | **Computationally Complex:** Requires matrix inversion ($\Sigma^{-1}$) and multiplication. |
| **Robust to Noise:** Inherently handles the correlated noise and scale differences in the state space. | **Requires State Estimator:** Cannot be used directly on raw UWB data; requires the UKF to provide the covariance matrix. |
| **Optimal for Fusion:** The mathematically correct metric for comparing an observation (camera tracklet) to a state estimate (UKF output). | **Tuning:** Requires careful tuning of the UKF's process and measurement noise parameters. |

### 4.2. Euclidean Distance

The Euclidean distance ($E$) is the simplest measure of distance between two points in space.

$$
E = \sqrt{(x_{cam} - x_{uwb})^2 + (y_{cam} - y_{uwb})^2}
$$

| Advantage | Disadvantage |
| :--- | :--- |
| **Simplicity:** Extremely fast and easy to compute. | **Non-Probabilistic:** Treats all errors equally, regardless of the UKF's confidence. |
| **No Covariance Required:** Can be used directly on raw position data. | **Sensitive to Noise:** Highly susceptible to instantaneous noise spikes in either the UWB or camera data. |
| **Scale Invariant:** (If coordinates are normalized) | **Sub-Optimal for Fusion:** Ignores the rich uncertainty information provided by the UKF. |

### 4.3. Dynamic Time Warping (DTW)

DTW is an algorithm used to measure the similarity between two temporal sequences that may vary in speed or timing.

| Advantage | Disadvantage |
| :--- | :--- |
| **Temporal Flexibility:** Excellent for matching trajectories that are similar in shape but misaligned in time (e.g., one person walks slower than the other). | **High Computational Cost:** $O(N^2)$ complexity, making it unsuitable for real-time, high-frequency matching. |
| **Shape Matching:** Focuses on the overall shape of the trajectory, ignoring minor temporal shifts. | **Overkill for Synchronized Data:** The UKF already synchronizes the data to 30 FPS, making DTW's time-warping capability largely redundant. |
| **Noisy Data Handling:** Can be robust to noise if a local distance metric (like Mahalanobis) is used within the DTW calculation. | **Complexity:** Difficult to integrate into a real-time assignment problem due to the variable-length path. |

### 4.4. Comparison Table

| Metric | Basis | Real-Time Suitability | Robustness to UKF Uncertainty | Primary Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Mahalanobis** | Probabilistic Distance | **High** (with sliding window) | **Excellent** | State Estimation Fusion |
| **Euclidean** | Geometric Distance | **Excellent** (Fastest) | **Poor** | Simple Proximity Check |
| **DTW** | Temporal Alignment | **Poor** (High $O(N^2)$ cost) | **Fair** | Misaligned Trajectory Shape Matching |

## 5. Conclusion

The final system design, utilizing the **Mahalanobis distance** within a **sliding window** and managed by a **Multi-Person Tracker**, represents the optimal balance between mathematical rigor, real-time performance, and robustness for the privacy-preserving Opt-in Camera application.

***

## References

[1] Original Research Paper: "Opt-in Camera: Person Identification in Video via UWB Localization and Its Application to Opt-in Systems"
[2] Mahalanobis, P. C. (1936). On the generalized distance in statistics. *Proceedings of the National Institute of Sciences of India*.
[3] Keogh, E., & Ratanamahatana, C. A. (2005). Exact indexing of dynamic time warping. *Knowledge and Information Systems*.
