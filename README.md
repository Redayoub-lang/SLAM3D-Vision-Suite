# SLAM3D-Vision-Suite: Visual-LiDAR 3D SLAM Engine

An industrial-grade 3D Simultaneous Localization and Mapping (SLAM) pipeline featuring $SE(3)$ Lie Group transformation estimation, Voxel Grid point cloud downsampling, Loop Closure Detection, and real-time mapping dashboards.

---

## Core Architecture & Features

* **6-DoF Motion Estimation:** Trajectory tracking represented in $SE(3)$ transformation space.
* **3D Voxel Grid Downsampling:** Discretization algorithm for spatial point cloud optimization.
* **Loop Closure Detection Engine:** Automatic spatial graph constraint matching for drift elimination.
* **Telemetry CSV Logger:** Automatic logging of 3D trajectory positions and mapping voxel counts to `slam_trajectory_log.csv`.

---

## Mathematical Formulation

Transformation between frame keyframes is computed as:

$$\mathbf{T} = \begin{bmatrix} \mathbf{R}_{3\times3} & \mathbf{t}_{3\times1} \\ \mathbf{0}_{1\times3} & 1 \end{bmatrix} \in SE(3)$$

---

## Quick Start

```bash
# Install requirements
pip install -r requirements.txt

# Launch SLAM System
python slam_3d_vision_engine.py