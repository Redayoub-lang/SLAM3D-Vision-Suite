# SLAM3D-Vision-Suite (v3.0 Enterprise Edition)
## Factor Graph Visual-LiDAR-Inertial SLAM with Semantic Dynamic Filtering

An enterprise-grade 3D Simultaneous Localization and Mapping (SLAM) suite featuring **6-DOF IMU Pre-Integration on $SO(3)$**, **Factor Graph Residual Optimization**, **AI Semantic Dynamic Point Cloud Masking**, and **Multi-Session Submap Graph Merging**.

---

## Enterprise Features in v3.0

* **IMU Pre-Integration Factor Graph:** Tight coupling of 6-DOF accelerometer and gyroscope biases with LiDAR geometric factors.
* **Semantic Dynamic Filtering:** Real-time identification and dynamic removal of moving objects (pedestrians, vehicles) from voxel reconstruction.
* **Multi-Session Submap Graph Manager:** Automated submap splitting, locking, and global keyframe optimization.
* **Tactical VIO HUD:** Displays live VIO trajectory, IMU pre-integration integration time, submap status, and filtered point metrics.
* **Telemetry Data Exporter:** Auto-exports factor graph VIO telemetry to `slam_v3_factor_graph_telemetry.csv`.

---

## Mathematical Formulation

IMU Pre-Integration delta rotation on $SO(3)$:

$$\Delta \mathbf{R}_{ij} = \prod_{k=i}^{j-1} \text{Exp}\left( (\boldsymbol{\omega}_k - \mathbf{b}_g) \Delta t \right)$$

Factor Graph Residual Minimization:

$$\arg\min_{\mathcal{X}} \left( \sum_{k} \Vert{}\mathbf{r}_{\text{prior}}\Vert{}^2 + \sum \Vert{}\mathbf{r}_{\text{IMU}}\Vert{}^2 + \sum \Vert{}\mathbf{r}_{\text{LiDAR}}\Vert{}^2 \right)$$

---

## Quick Execution

```bash
# Install dependencies
pip install -r requirements.txt

# Run v3.0 Engine
python slam3d_vision_engine.py