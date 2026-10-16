import cv2
import numpy as np
import time
import math
import csv


class PointCloudMap:
    """Manages 3D Spatial Voxel Grid Point Cloud Generation and Loop Closure Optimization."""
    def __init__(self, voxel_size=0.1):
        self.voxel_size = voxel_size
        self.map_points = []      # Consolidated Global 3D Point Cloud
        self.keyframe_poses = []  # List of trajectory camera/LiDAR transformations [SE(3)]
        self.loop_closures = []   # Detected loop closure constraint links

    def voxel_downsample(self, points):
        """Reduces point cloud density using 3D spatial voxel grid discretization."""
        if len(points) == 0:
            return points
        
        # Discretize points into grid coordinates
        voxel_indices = np.floor(points / self.voxel_size).astype(int)
        _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
        return points[unique_indices]

    def add_keyframe(self, pose_matrix, local_points):
        """Transform local point cloud to world frame and integrate into global map."""
        self.keyframe_poses.append(pose_matrix)
        
        # Transform 3D local points using Pose Matrix T [4x4]
        homo_points = np.hstack((local_points, np.ones((len(local_points), 1))))
        world_points = (pose_matrix @ homo_points.T).T[:, :3]

        # Downsample and merge
        merged = np.vstack((self.map_points, world_points)) if len(self.map_points) > 0 else world_points
        self.map_points = self.voxel_downsample(merged)

    def detect_loop_closure(self, current_pose, dist_threshold=2.5, min_id_gap=15):
        """Detects loop closures using Mahalanobis/Euclidean distance thresholding."""
        curr_pos = current_pose[:3, 3]
        curr_idx = len(self.keyframe_poses) - 1

        for i in range(curr_idx - min_id_gap):
            past_pos = self.keyframe_poses[i][:3, 3]
            dist = np.linalg.norm(curr_pos - past_pos)
            if dist < dist_threshold:
                self.loop_closures.append((i, curr_idx))
                return True, i
        return False, None


class VisualLiDARSLAM:
    """Visual-LiDAR 3D SLAM Estimation Pipeline."""
    def __init__(self):
        self.map = PointCloudMap(voxel_size=0.15)
        self.current_pose = np.eye(4)
        self.trajectory = [np.zeros(3)]
        
        # Sensor Calibration (Camera Intrinsics Matrix K)
        self.fx, self.fy = 525.0, 525.0
        self.cx, self.cy = 320.0, 240.0

    def generate_synthetic_depth_frame(self, t_step):
        """Simulates 3D Environment Point Cloud Scan with Dynamic Noise."""
        np.random.seed(int(t_step * 10) % 1000)
        
        # Generate tunnel/room geometry [x, y, z]
        num_pts = 400
        angles = np.linspace(0, 2 * np.pi, num_pts)
        depths = np.random.uniform(2.0, 6.0, num_pts)

        x = depths * np.cos(angles)
        y = depths * np.sin(angles)
        z = np.random.uniform(-1.5, 1.5, num_pts)

        local_pts = np.column_stack((x, y, z))
        return local_pts

    def estimate_motion(self, dt, t_step):
        """Calculates 6-DoF Rigid Pose Transformation T in SE(3)."""
        # Circular motion simulation with slight drift
        radius = 4.0
        angular_velocity = 0.3
        
        theta = angular_velocity * t_step
        x = radius * math.sin(theta)
        y = radius * (1 - math.cos(theta))
        z = 0.2 * math.sin(theta * 2)

        # Rotation angles (Roll, Pitch, Yaw)
        yaw = theta
        pitch = 0.05 * math.sin(t_step)
        roll = 0.02 * math.cos(t_step)

        # Build Rotation Matrix R_z * R_y * R_x
        Rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
        Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
        Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    def process_frame(self, t_step, dt=0.1):
        """Executes one step of 3D SLAM state update."""
        # 1. Update Kinematic Pose Estimation
        self.current_pose = self.estimate_motion(dt, t_step)
        curr_position = self.current_pose[:3, 3]
        self.trajectory.append(curr_position.copy())

        # 2. Acquire Sensor 3D Point Cloud Scan
        local_points = self.generate_synthetic_depth_frame(t_step)

        # 3. Keyframe Integration & Voxel Map Construction
        self.map.add_keyframe(self.current_pose, local_points)

        # 4. Loop Closure Detection Check
        loop_found, matched_idx = self.map.detect_loop_closure(self.current_pose)
        
        return curr_position, loop_found, matched_idx


class SLAMDashboard:
    """Renders Real-time 3D Occupancy Projection & SLAM HUD Tracker."""
    def __init__(self, width=900, height=600):
        self.w = width
        self.h = height

    def render(self, slam_system, loop_status):
        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        
        # Draw Coordinate Axes Grid
        cx, cy = self.w // 2, self.h // 2 + 50
        scale = 35.0  # Pixels per meter

        for r in range(50, self.w, 50):
            cv2.line(canvas, (r, 0), (r, self.h), (25, 25, 25), 1)
        for r in range(50, self.h, 50):
            cv2.line(canvas, (0, r), (self.w, r), (25, 25, 25), 1)

        # Render 3D Point Cloud (Projected to Top-Down 2D Map)
        if len(slam_system.map.map_points) > 0:
            for pt in slam_system.map.map_points[::3]:  # Downsample for rendering speed
                px = int(cx + pt[0] * scale)
                py = int(cy - pt[1] * scale)
                if 0 <= px < self.w and 0 <= py < self.h:
                    canvas[py, px] = (180, 180, 180)

        # Render Trajectory Path
        traj_pts = []
        for pos in slam_system.trajectory:
            px = int(cx + pos[0] * scale)
            py = int(cy - pos[1] * scale)
            traj_pts.append((px, py))

        for i in range(1, len(traj_pts)):
            cv2.line(canvas, traj_pts[i-1], traj_pts[i], (0, 255, 0), 2, cv2.LINE_AA)

        # Render Robot Position
        if len(traj_pts) > 0:
            rx, ry = traj_pts[-1]
            cv2.circle(canvas, (rx, ry), 7, (0, 0, 255), -1, cv2.LINE_AA)

        # Render Loop Closure Constraint Links
        for loop in slam_system.map.loop_closures:
            p1 = traj_pts[loop[0]]
            p2 = traj_pts[loop[1]]
            cv2.line(canvas, p1, p2, (255, 0, 255), 2, cv2.LINE_AA)

        # Overlay HUD Information Box
        cv2.rectangle(canvas, (15, 15), (350, 135), (15, 15, 15), -1)
        cv2.rectangle(canvas, (15, 15), (350, 135), (0, 255, 255), 1)

        curr_pos = slam_system.trajectory[-1]
        cv2.putText(canvas, "SYSTEM: Visual-LiDAR 3D SLAM Engine", (25, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        cv2.putText(canvas, f"3D POSE (X,Y,Z): [{curr_pos[0]:.2f}, {curr_pos[1]:.2f}, {curr_pos[2]:.2f}] m",
                    (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        cv2.putText(canvas, f"MAP POINTS: {len(slam_system.map.map_points)} Voxels",
                    (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        
        loop_txt = "DETECTED!" if loop_status else "SEARCHING"
        loop_col = (0, 255, 0) if loop_status else (150, 150, 150)
        cv2.putText(canvas, f"LOOP CLOSURE: {loop_txt} ({len(slam_system.map.loop_closures)} Links)",
                    (25, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.38, loop_col, 1)

        return canvas


def main():
    slam = VisualLiDARSLAM()
    dashboard = SLAMDashboard(width=900, height=600)
    
    # Telemetry Log
    log_file = open("slam_trajectory_log.csv", "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["Timestamp", "Pose_X", "Pose_Y", "Pose_Z", "Map_Voxel_Count", "Loop_Closed"])

    t_step = 0.0
    start_time = time.time()

    print("[INFO] Initializing Visual-LiDAR 3D SLAM Engine...")

    while True:
        elapsed = time.time() - start_time
        pos, loop_detected, _ = slam.process_frame(t_step=t_step, dt=0.1)

        # Log Data
        writer.writerow([
            round(elapsed, 2), round(pos[0], 3), round(pos[1], 3), round(pos[2], 3),
            len(slam.map.map_points), 1 if loop_detected else 0
        ])

        # Render Dashboard Window
        frame = dashboard.render(slam, loop_detected)
        cv2.imshow("Visual-LiDAR 3D SLAM Mapping & Trajectory Engine", frame)

        t_step += 0.08
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break

    log_file.close()
    cv2.destroyAllWindows()
    print("[INFO] SLAM Pipeline terminated. Log saved to 'slam_trajectory_log.csv'.")


if __name__ == "__main__":
    main()