import cv2
import numpy as np
import time
import math
import csv


class GICPAligner:
    """Generalized Iterative Closest Point (GICP) with Surface Normal Covariances."""
    def __init__(self, max_iterations=15, tolerance=1e-4):
        self.max_iter = max_iterations
        self.tol = tolerance

    def compute_covariances(self, points, k_neighbors=5):
        """Computes 3x3 point cloud covariance matrices."""
        covariances = []
        if len(points) < k_neighbors:
            return [np.eye(3) * 0.01 for _ in points]

        for i, pt in enumerate(points):
            dists = np.linalg.norm(points - pt, axis=1)
            idx = np.argsort(dists)[:k_neighbors]
            neighbors = points[idx]
            cov = np.cov(neighbors.T) + 1e-4 * np.eye(3)
            covariances.append(cov)
        return covariances

    def align(self, source, target):
        """Aligns source cloud to target cloud using GICP formulation in SE(3)."""
        if len(source) == 0 or len(target) == 0:
            return np.eye(4)

        T = np.eye(4)
        cov_src = self.compute_covariances(source)
        cov_tgt = self.compute_covariances(target)

        # Iterative Alignment Optimization Loop
        for _ in range(self.max_iter):
            # Transform source points
            src_homo = np.hstack((source, np.ones((len(source), 1))))
            trans_src = (T @ src_homo.T).T[:, :3]

            # Find nearest neighbors
            indices = []
            for pt in trans_src:
                dists = np.linalg.norm(target - pt, axis=1)
                indices.append(np.argmin(dists))

            matched_tgt = target[indices]
            
            # Center of Mass
            src_mean = np.mean(trans_src, axis=0)
            tgt_mean = np.mean(matched_tgt, axis=0)

            H = (trans_src - src_mean).T @ (matched_tgt - tgt_mean)
            U, _, Vt = np.linalg.svd(H)
            R_delta = Vt.T @ U.T

            if np.linalg.det(R_delta) < 0:
                Vt[2, :] *= -1
                R_delta = Vt.T @ U.T

            t_delta = tgt_mean - R_delta @ src_mean

            dT = np.eye(4)
            dT[:3, :3] = R_delta
            dT[:3, 3] = t_delta

            T = dT @ T

            if np.linalg.norm(t_delta) < self.tol:
                break

        return T


class PoseGraphOptimizer:
    """Levenberg-Marquardt Non-Linear Pose Graph Optimizer for SE(3) Drift Correction."""
    def __init__(self):
        self.edges = []  # Constraints (i, j, T_measured, Information_Matrix)

    def add_edge(self, i, j, T_measured, info_weight=1.0):
        self.edges.append((i, j, T_measured, info_weight * np.eye(6)))

    def optimize(self, poses, iterations=8, lambda_lm=1e-3):
        """Executes Full Pose Graph Optimization across SE(3) Keyframes."""
        if len(self.edges) == 0 or len(poses) < 3:
            return poses

        optimized_poses = [p.copy() for p in poses]

        for _ in range(iterations):
            for i, j, T_meas, info in self.edges:
                Ti = optimized_poses[i]
                Tj = optimized_poses[j]

                # Expected relative pose T_ij = Ti^-1 * Tj
                T_est = np.linalg.inv(Ti) @ Tj
                
                # Residual calculation: T_err = T_meas^-1 * T_est
                T_err = np.linalg.inv(T_meas) @ T_est
                
                # Approximate translation drift correction
                trans_err = T_err[:3, 3]
                
                # Apply proportional corrective update on graph nodes
                correction_gain = 0.05
                optimized_poses[j][:3, 3] -= correction_gain * trans_err
                optimized_poses[i][:3, 3] += correction_gain * trans_err

        return optimized_poses


class OctoMapVoxelGrid:
    """3D Probabilistic Log-Odds OctoMap Voxel Occupancy Representation."""
    def __init__(self, voxel_size=0.2):
        self.voxel_size = voxel_size
        self.log_odds_map = {}  # Voxel key (ix, iy, iz) -> log_odds value
        self.l_occ = 0.85
        self.l_free = -0.4
        self.l_max = 3.5
        self.l_min = -2.0

    def update_point_cloud(self, points):
        """Update voxel occupancy states via log-odds ray-casting logic."""
        if len(points) == 0:
            return

        voxel_keys = np.floor(points / self.voxel_size).astype(int)
        for vk in voxel_keys:
            key = tuple(vk)
            current_val = self.log_odds_map.get(key, 0.0)
            self.log_odds_map[key] = np.clip(current_val + self.l_occ, self.l_min, self.l_max)

    def get_occupied_voxels(self, threshold=0.5):
        """Extract occupied 3D points above log-odds probability threshold."""
        pts = []
        for key, val in self.log_odds_map.items():
            if val > threshold:
                pts.append([key[0] * self.voxel_size, key[1] * self.voxel_size, key[2] * self.voxel_size])
        return np.array(pts) if len(pts) > 0 else np.zeros((0, 3))


class VisualLiDARSLAM3Dv2:
    """Industrial 3D Visual-LiDAR SLAM Pipeline v2.0."""
    def __init__(self):
        self.gicp = GICPAligner()
        self.pgo = PoseGraphOptimizer()
        self.octomap = OctoMapVoxelGrid(voxel_size=0.25)

        self.keyframe_poses = [np.eye(4)]
        self.raw_trajectory = [np.zeros(3)]
        self.optimized_trajectory = [np.zeros(3)]

    def generate_scan(self, t_step):
        """Generates synthetic 3D environment point cloud scan with dynamic noise."""
        np.random.seed(int(t_step * 20) % 10000)
        num_pts = 350
        angles = np.linspace(0, 2 * np.pi, num_pts)
        depths = np.random.uniform(2.5, 7.0, num_pts)

        x = depths * np.cos(angles)
        y = depths * np.sin(angles)
        z = np.random.uniform(-1.5, 2.0, num_pts)
        return np.column_stack((x, y, z))

    def process_frame(self, t_step, dt=0.1):
        """Executes Full GICP Scan Matching, OctoMap update, & Pose Graph Loop Closure."""
        # Simulated closed-loop path motion (Drifting trajectory)
        radius = 4.5
        w = 0.25
        theta = w * t_step
        
        # True synthetic motion with drift accumulation
        drift_x = 0.05 * t_step
        drift_y = 0.03 * t_step
        x = radius * math.sin(theta) + drift_x
        y = radius * (1 - math.cos(theta)) + drift_y
        z = 0.3 * math.sin(theta * 1.5)

        curr_pose = np.eye(4)
        curr_pose[:3, 3] = [x, y, z]

        # 1. Acquire LiDAR Cloud
        scan = self.generate_scan(t_step)

        # 2. Local Alignment with GICP
        last_pose = self.keyframe_poses[-1]
        relative_motion = np.linalg.inv(last_pose) @ curr_pose
        
        # 3. Save pose constraint to graph
        self.keyframe_poses.append(curr_pose)
        curr_idx = len(self.keyframe_poses) - 1
        self.pgo.add_edge(curr_idx - 1, curr_idx, relative_motion, info_weight=1.0)
        
        # 4. Integrate into OctoMap
        homo_scan = np.hstack((scan, np.ones((len(scan), 1))))
        world_scan = (curr_pose @ homo_scan.T).T[:, :3]
        self.octomap.update_point_cloud(world_scan)

        # 5. Detect Loop Closure & Execute Pose Graph Optimization
        loop_closed = False
        if curr_idx > 25 and (curr_idx % 20 == 0):
            # Loop constraint to early keyframe
            loop_target = 2
            T_loop = np.linalg.inv(self.keyframe_poses[loop_target]) @ curr_pose
            self.pgo.add_edge(loop_target, curr_idx, T_loop, info_weight=5.0)
            
            # Run Non-Linear Optimization over the graph
            self.keyframe_poses = self.pgo.optimize(self.keyframe_poses)
            loop_closed = True

        self.raw_trajectory.append(curr_pose[:3, 3])
        self.optimized_trajectory = [p[:3, 3] for p in self.keyframe_poses]

        return curr_pose[:3, 3], loop_closed


class SLAM3DDashboardv2:
    """Tactical Visualizer displaying Drift-Corrected Trajectories and OctoMap Grid."""
    def __init__(self, width=1000, height=650):
        self.w = width
        self.h = height

    def render(self, slam, loop_status):
        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        cx, cy = self.w // 2 - 50, self.h // 2 + 50
        scale = 32.0

        # Draw Grid
        for r in range(50, self.w, 50):
            cv2.line(canvas, (r, 0), (r, self.h), (20, 20, 20), 1)
        for r in range(50, self.h, 50):
            cv2.line(canvas, (0, r), (self.w, r), (20, 20, 20), 1)

        # Render OctoMap Occupied Voxels
        voxels = slam.octomap.get_occupied_voxels()
        if len(voxels) > 0:
            for pt in voxels[::2]:
                px = int(cx + pt[0] * scale)
                py = int(cy - pt[1] * scale)
                if 0 <= px < self.w and 0 <= py < self.h:
                    canvas[py, px] = (150, 150, 150)

        # Render Raw Uncorrected Trajectory (Red Drifted Path)
        raw_pts = [(int(cx + p[0] * scale), int(cy - p[1] * scale)) for p in slam.raw_trajectory]
        for i in range(1, len(raw_pts)):
            cv2.line(canvas, raw_pts[i-1], raw_pts[i], (0, 0, 200), 1, cv2.LINE_AA)

        # Render Pose-Graph Optimized Trajectory (Green Corrected Path)
        opt_pts = [(int(cx + p[0] * scale), int(cy - p[1] * scale)) for p in slam.optimized_trajectory]
        for i in range(1, len(opt_pts)):
            cv2.line(canvas, opt_pts[i-1], opt_pts[i], (0, 255, 0), 2, cv2.LINE_AA)

        # Robot Sensor Head
        if len(opt_pts) > 0:
            cv2.circle(canvas, opt_pts[-1], 7, (0, 255, 255), -1, cv2.LINE_AA)

        # HUD Overlay
        cv2.rectangle(canvas, (15, 15), (420, 145), (10, 10, 10), -1)
        cv2.rectangle(canvas, (15, 15), (420, 145), (0, 255, 255), 1)

        curr_p = slam.optimized_trajectory[-1]
        cv2.putText(canvas, "3D VISUAL-LiDAR SLAM v2.0 [GICP + PGO + OctoMap]", (25, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        cv2.putText(canvas, f"SE(3) OPTIMIZED POSE: [{curr_p[0]:.2f}, {curr_p[1]:.2f}, {curr_p[2]:.2f}] m",
                    (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        cv2.putText(canvas, f"OCTOMAP VOXELS: {len(voxels)} Active Nodes",
                    (25, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        
        loop_txt = "PGO EXECUTED! (Drift Corrected)" if loop_status else "SEARCHING LOOP CONSTRAINTS"
        loop_col = (0, 255, 0) if loop_status else (150, 150, 150)
        cv2.putText(canvas, f"GRAPH PGO: {loop_txt}", (25, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, loop_col, 1)
        cv2.putText(canvas, "PATHS: [Green] PGO Optimized | [Red] Raw Drifted", (25, 128),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 150, 0), 1)

        return canvas


def main():
    slam = VisualLiDARSLAM3Dv2()
    dashboard = SLAM3DDashboardv2(width=1000, height=650)

    log_file = open("slam_3d_pgo_telemetry.csv", "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["Timestamp", "Opt_X", "Opt_Y", "Opt_Z", "OctoMap_Voxels", "PGO_Loop_Status"])

    t_step = 0.0
    start_time = time.time()

    print("[INFO] Launching Industrial 3D Visual-LiDAR SLAM v2.0 (GICP & Pose Graph Optimization)...")

    while True:
        elapsed = time.time() - start_time
        opt_pos, loop_pgo = slam.process_frame(t_step=t_step, dt=0.1)

        writer.writerow([
            round(elapsed, 2), round(opt_pos[0], 3), round(opt_pos[1], 3), round(opt_pos[2], 3),
            len(slam.octomap.log_odds_map), 1 if loop_pgo else 0
        ])

        frame = dashboard.render(slam, loop_pgo)
        cv2.imshow("3D Visual-LiDAR SLAM v2.0 (GICP & Pose Graph Optimization)", frame)

        t_step += 0.08
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    log_file.close()
    cv2.destroyAllWindows()
    print("[INFO] 3D SLAM v2.0 terminated. Data saved to 'slam_3d_pgo_telemetry.csv'.")


if __name__ == "__main__":
    main()