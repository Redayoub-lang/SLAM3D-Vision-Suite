# SLAM3D-Vision-Suite (v2.0 Industrial Edition)
## Visual-LiDAR 3D SLAM with GICP, Pose Graph Optimization (PGO) & OctoMap

An industrial-grade 3D Simultaneous Localization and Mapping (SLAM) pipeline featuring **Generalized ICP (GICP)** scan matching, **SE(3) Levenberg-Marquardt Pose Graph Optimization (PGO)**, and **3D Probabilistic Log-Odds OctoMap Voxel Mapping**.

---

## Core Features in v2.0

* **Generalized ICP (GICP):** Surface normal covariance alignment in $SE(3)$ for high-precision frame registration.
* **Non-Linear Pose Graph Optimization (PGO):** Corrects global trajectory drift using Levenberg-Marquardt optimization upon loop closure detection.
* **Probabilistic 3D OctoMap:** Memory-efficient voxel grid log-odds representation for 3D navigation.
* **Dual Trajectory Dashboard:** Live visual comparison between uncorrected raw drift trajectory (Red) and PGO optimized trajectory (Green).
* **Data Exporter:** Auto-logging pose optimization parameters to `slam_3d_pgo_telemetry.csv`.

---

## Mathematical Formulation

Pose graph edge residual error $\mathbf{e}_{ij} \in \mathbb{R}^6$ in $SE(3)$ is given by:

$$\mathbf{e}_{ij}(\mathbf{T}_i, \mathbf{T}_j) = \text{Log}_{SE(3)} \left( \left(\mathbf{T}_{ij}^{\text{meas}}\right)^{-1} \cdot \mathbf{T}_i^{-1} \cdot \mathbf{T}_j \right)^\vee$$

Optimized via Levenberg-Marquardt update:

$$\left( \mathbf{J}^T \mathbf{\Omega} \mathbf{J} + \lambda \text{diag}(\mathbf{J}^T \mathbf{\Omega} \mathbf{J}) \right) \Delta \mathbf{x} = -\mathbf{J}^T \mathbf{\Omega} \mathbf{e}$$

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch 3D SLAM Engine
python slam3d_vision_engine.py