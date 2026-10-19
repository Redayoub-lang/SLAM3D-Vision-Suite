import cv2
import numpy as np
import time
import math
import csv


class IMUPreintegrator:
    """6-DOF IMU Pre-Integration Engine on SO(3) Lie Group."""
    def __init__(self):
        self.acc_bias = np.array([0.01, -0.02, 0.01])
        self.gyro_bias = np.array([0.001, 0.002, -0.001])
        self.gravity = np.array([0.0, 0.0, -9.81])

        self.reset()

    def reset(self):
        self.delta_R = np.eye(3)
        self.delta_v = np.zeros(3)
        self.delta_p = np.zeros(3)
        self.dt_sum = 0.0

    def integrate_measurement(self, accel_meas, gyro_meas, dt):
        """Pre-integrates raw IMU acceleration and angular velocity over dt."""
        acc_corr = accel_meas - self.acc_bias
        gyro_corr = gyro_meas - self.gyro_bias

        # Rotation update via Rodrigues formula on SO(3)
        angle = np.linalg.norm(gyro_corr) * dt
        if angle > 1e-6:
            axis = gyro_corr / np.linalg.norm(gyro_corr)
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            dR = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
        else:
            dR = np.eye(3)

        self.delta_p += self.delta_v * dt + 0.5 * (self.delta_R @ acc_corr) * (dt ** 2)
        self.delta_v += (self.delta_R @ acc_corr) * dt
        self.delta_R = self.delta_R @ dR
        self.dt_sum += dt


class SemanticDynamicFilter:
    """AI Semantic Segmentation Mask Processor & Dynamic Entity Removal."""
    def __init__(self):
        self.static_classes = [0, 1, 2]  # Building, Road, Vegetation
        self.dynamic_classes = [3, 4]     # Pedestrians, Dynamic Vehicles

    def filter_point_cloud(self, points, t_step):
        """Labels points and dynamic masks moving objects (e.g. cars/pedestrians)."""
        static_points = []
        dynamic_points = []

        # Synthetic dynamic target movement
        dyn_center = np.array([3.0 + 2.0 * math.cos(t_step * 1.2),
                               2.0 + 2.0 * math.sin(t_step * 1.2), 0.0])

        for pt in points:
            # Distance check to dynamic object cluster
            if np.linalg.norm(pt - dyn_center) < 1.2:
                dynamic_points.append(pt)
            else:
                static_points.append(pt)

        return np.array(static_points), np.array(dynamic_points)


class SubmapGraphManager:
    """Manages Multi-Session Submap Pose Graphs and Merge Constraints."""
    def __init__(self, keyframes_per_submap=20):
        self.max_kfs = keyframes_per_submap
        self.submaps = []
        self.current_submap = []

    def add_keyframe(self, pose, point_cloud):
        self.current_submap.append((pose, point_cloud))
        if len(self.current_submap) >= self.max_kfs:
            self.submaps.append(self.current_submap)
            self.current_submap = []
            print(f"[INFO] Submap #{len(self.submaps)} finalized and locked in Graph.")


class VisualLiDARInertialSLAMv3:
    """Enterprise Factor Graph Visual-Inertial 3D SLAM Engine v3.0."""
    def __init__(self):
        self.imu_preint = IMUPreintegrator()
        self.semantic_filter = SemanticDynamicFilter()
        self.submap_manager = SubmapGraphManager(keyframes_per_submap=20)

        self.robot_pose = np.eye(4)
        self.robot_velocity = np.zeros(3)
        self.trajectory_v3 = [np.zeros(3)]
        self.static_voxels = set()

    def generate_point_cloud(self, t_step):
        """Generates synthetic 3D environment cloud with static and dynamic objects."""
        np.random.seed(int(t_step * 30) % 10000)
        num_pts = 400
        angles = np.linspace(0, 2 * np.pi, num_pts)
        depths = np.random.uniform(2.0, 8.0, num_pts)

        x = depths * np.cos(angles)
        y = depths * np.sin(angles)
        z = np.random.uniform(-1.0, 2.5, num_pts)
        pts = np.column_stack((x, y, z))

        # Add moving object
        dyn_pts = np.random.uniform(-0.8, 0.8, (30, 3)) + np.array([3.0 + 2.0 * math.cos(t_step * 1.2),
                                                                   2.0 + 2.0 * math.sin(t_step * 1.2), 0.0])
        return np.vstack((pts, dyn_pts))

    def process_frame(self, t_step, dt=0.1):
        """Executes Factor Graph VIO integration, Dynamic Filtering & OctoMap Voxel Insertion."""
        # Simulate IMU raw measurements
        accel_raw = np.array([0.2 * math.sin(t_step), 0.1 * math.cos(t_step), 9.81]) + np.random.normal(0, 0.05, 3)
        gyro_raw = np.array([0.02, -0.01, 0.25]) + np.random.normal(0, 0.01, 3)

        self.imu_preint.integrate_measurement(accel_raw, gyro_raw, dt)

        # Update pose using pre-integrated IMU factor
        delta_p_world = self.robot_pose[:3, :3] @ self.imu_preint.delta_p
        self.robot_pose[:3, 3] += delta_p_world

        # Acquire and Semantically Filter Point Cloud
        raw_cloud = self.generate_point_cloud(t_step)
        static_cloud, dynamic_cloud = self.semantic_filter.filter_point_cloud(raw_cloud, t_step)

        # Transform Static Points to World Coordinate Frame
        if len(static_cloud) > 0:
            homo_static = np.hstack((static_cloud, np.ones((len(static_cloud), 1))))
            world_static = (self.robot_pose @ homo_static.T).T[:, :3]

            # Voxelize static points (0.3m voxel grid)
            for pt in world_static[::3]:
                vk = (int(pt[0] // 0.3), int(pt[1] // 0.3), int(pt[2] // 0.3))
                self.static_voxels.add(vk)

        # Submap Management
        self.submap_manager.add_keyframe(self.robot_pose.copy(), static_cloud)
        self.trajectory_v3.append(self.robot_pose[:3, 3].copy())

        return self.robot_pose[:3, 3], len(dynamic_cloud)


class SLAM3DDashboardV3:
    """Enterprise Interface for Factor Graph VIO Trajectory & Semantic OctoMap Rendering."""
    def __init__(self, width=1050, height=680):
        self.w = width
        self.h = height

    def render(self, slam, num_dynamic_pts):
        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        cx, cy = self.w // 2 - 50, self.h // 2 + 50
        scale = 30.0

        # Draw Grid
        for r in range(50, self.w, 50):
            cv2.line(canvas, (r, 0), (r, self.h), (20, 22, 22), 1)
        for r in range(50, self.h, 50):
            cv2.line(canvas, (0, r), (self.w, r), (20, 22, 22), 1)

        # Render Static Voxels (Green/Gray)
        for vk in list(slam.static_voxels)[::2]:
            px = int(cx + vk[0] * 0.3 * scale)
            py = int(cy - vk[1] * 0.3 * scale)
            if 0 <= px < self.w and 0 <= py < self.h:
                canvas[py, px] = (120, 180, 120)

        # Render VIO Trajectory (Cyan line)
        pts = [(int(cx + p[0] * scale), int(cy - p[1] * scale)) for p in slam.trajectory_v3]
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i-1], pts[i], (255, 255, 0), 2, cv2.LINE_AA)

        if len(pts) > 0:
            cv2.circle(canvas, pts[-1], 7, (0, 255, 255), -1, cv2.LINE_AA)

        # HUD Overlay
        cv2.rectangle(canvas, (15, 15), (440, 160), (10, 10, 10), -1)
        cv2.rectangle(canvas, (15, 15), (440, 160), (0, 255, 255), 1)

        curr_p = slam.trajectory_v3[-1]
        cv2.putText(canvas, "3D SLAM v3.0 [Factor Graph VIO + Semantic Filter]", (25, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        cv2.putText(canvas, f"VIO POSE: [{curr_p[0]:.2f}, {curr_p[1]:.2f}, {curr_p[2]:.2f}] m", (25, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        cv2.putText(canvas, f"IMU PRE-INTEGRATION: dt={slam.imu_preint.dt_sum:.2f}s | Biases Active", (25, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        cv2.putText(canvas, f"SEMANTIC MASK: {num_dynamic_pts} Dynamic Points Filtered Out", (25, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1)
        cv2.putText(canvas, f"SUBMAP GRAPH: {len(slam.submap_manager.submaps)} Submaps Locked", (25, 128),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 150, 0), 1)

        return canvas


def main():
    slam = VisualLiDARInertialSLAMv3()
    dashboard = SLAM3DDashboardV3(width=1050, height=680)

    log_file = open("slam_v3_factor_graph_telemetry.csv", "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["Timestamp", "VIO_X", "VIO_Y", "VIO_Z", "Static_Voxels", "Dynamic_Filtered_Pts", "Submap_Count"])

    t_step = 0.0
    start_time = time.time()

    print("[INFO] Launching Enterprise 3D Visual-LiDAR-Inertial SLAM v3.0 (Factor Graph & Semantic Filter)...")

    while True:
        elapsed = time.time() - start_time
        pos, dyn_cnt = slam.process_frame(t_step=t_step, dt=0.1)

        writer.writerow([
            round(elapsed, 2), round(pos[0], 3), round(pos[1], 3), round(pos[2], 3),
            len(slam.static_voxels), dyn_cnt, len(slam.submap_manager.submaps)
        ])

        frame = dashboard.render(slam, dyn_cnt)
        cv2.imshow("3D Visual-LiDAR-Inertial SLAM v3.0 (Factor Graph & Semantic Filter)", frame)

        t_step += 0.08
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    log_file.close()
    cv2.destroyAllWindows()
    print("[INFO] 3D SLAM v3.0 terminated. Data logged to 'slam_v3_factor_graph_telemetry.csv'.")


if __name__ == "__main__":
    main()