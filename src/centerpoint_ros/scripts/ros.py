#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import threading

# Make PyTorch's bundled shared libraries loadable
torch_lib = "/usr/local/lib/python{}.{}/dist-packages/torch/lib".format(
    sys.version_info.major, sys.version_info.minor)
if os.path.exists(torch_lib):
    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if torch_lib not in ld_path:
        os.environ['LD_LIBRARY_PATH'] = f"{torch_lib}:{ld_path}"

import numpy as np
import torch
from pyquaternion import Quaternion

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

# Package source root (override with CENTERPOINT_ROS_ROOT if cloned elsewhere)
PKG_ROOT = os.environ.get(
    "CENTERPOINT_ROS_ROOT", "/home/centerpoint_ws/src/centerpoint_ros")

# Make det3d importable
sys.path.insert(0, PKG_ROOT)
from det3d.torchie import Config
from det3d.models import build_detector
from det3d.torchie.trainer import load_checkpoint
from det3d.datasets.pipelines import Compose


# =============================================================
# Settings - this is the only place you need to touch for tuning.
# =============================================================
# Topics
INPUT_TOPIC = '/pointcloud/vlp16'
OUTPUT_TOPIC_VISION = '/detections_vision'
OUTPUT_TOPIC_MARKERS = '/detections_markers'

# QoS / inference loop
QOS_DEPTH = 100
QOS_RELIABILITY = QoSReliabilityPolicy.RELIABLE   # or BEST_EFFORT
INFERENCE_RATE = 10          # Hz

# Detection confidence threshold (0.0 ~ 1.0)
SCORE_THRESHOLD = 0.3

# Point cloud preprocessing
PC_MAX_RANGE = 100.0         # max detection range in meters
PC_MIN_POINTS = 30           # minimum number of points to run inference
PC_Z_MIN = -2.0              # ground filter
PC_Z_MAX = 4.0               # height limit
PC_R_MIN = 0.5               # drop noise near the sensor

# Intensity normalization
INTENSITY_MAX = 9999.0

# BBox filtering (a box passes when dx, dy, dz are all above this)
BBOX_MIN_SIZE = 0.0

# NMS (Non-Maximum Suppression)
NMS_IOU_THRESHOLD = 0.4      # 0.3 ~ 0.7 recommended
NMS_ENABLE = False

# Debug logging
DEBUG_ENABLE = False
# =============================================================


# Keep these as module-level helpers, with the defaults spelled out.
def sanitize_points(np_p, has_intensity, max_range=PC_MAX_RANGE, min_points=PC_MIN_POINTS,
                   z_min=PC_Z_MIN, z_max=PC_Z_MAX, r_min=PC_R_MIN, intensity_max=INTENSITY_MAX):

    if np_p is None or np_p.ndim != 2 or np_p.shape[1] < 3:
        return None

    np_p = np_p.astype(np.float32, copy=False)
    finite_mask = np.isfinite(np_p[:, 0]) & np.isfinite(np_p[:, 1]) & np.isfinite(np_p[:, 2])
    if has_intensity and np_p.shape[1] >= 4:
        finite_mask &= np.isfinite(np_p[:, 3])
    np_p = np_p[finite_mask]
    if np_p.shape[0] == 0:
        return None

    r = np.linalg.norm(np_p[:, :3], axis=1)
    z_ok = (np_p[:, 2] > z_min) & (np_p[:, 2] < z_max)
    r_ok = (r > r_min) & (r < max_range)
    mask = z_ok & r_ok
    np_p = np_p[mask]
    if np_p.shape[0] < min_points:
        return None

    if has_intensity and np_p.shape[1] >= 4:
        i = np_p[:, 3].astype(np.float32)
        i = np.nan_to_num(i, nan=0.0, posinf=0.0, neginf=0.0)
        i = np.maximum(i, 0.0)
        i = np.clip(i, 0.0, intensity_max)
        i = i / intensity_max
    else:
        i = np.zeros((np_p.shape[0],), dtype=np.float32)

    elongation = np.zeros((np_p.shape[0],), dtype=np.float32)
    points = np.stack((np_p[:, 0], np_p[:, 1], np_p[:, 2], i, elongation), axis=1)
    return points[np.isfinite(points).all(axis=1)]


class SafeRate:
    def __init__(self, node, hz):
        self.node = node
        self.hz = hz
        self.period = 1.0 / hz

    def sleep(self):
        time.sleep(self.period)


class CenterPoint_Det3D_ROS(Node):
    def __init__(self):
        super().__init__('centerpoint_ros_node')

        self.latest_msg = None
        self.lock = threading.Lock()
        config_path, ckpt_path = self.init_ros()
        self.init_centerpoint(config_path, ckpt_path)

    def init_ros(self):
        self.declare_parameter('config_path',
            os.path.join(PKG_ROOT,
                "configs/waymo/voxelnet/two_stage/waymo_centerpoint_voxelnet_two_stage_bev_5point_ft_6epoch_freeze.py"))
        self.declare_parameter('ckpt_path',
            os.path.join(PKG_ROOT, "epoch_6.pth"))

        config_path = self.get_parameter('config_path').value
        ckpt_path = self.get_parameter('ckpt_path').value

        qos_profile = QoSProfile(
            reliability=QOS_RELIABILITY,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=QOS_DEPTH
        )

        self.sub_velo = self.create_subscription(
            PointCloud2,
            INPUT_TOPIC,
            self.lidar_callback,
            qos_profile
        )
        self.pub_vision = self.create_publisher(
            Detection3DArray, OUTPUT_TOPIC_VISION, QOS_DEPTH)
        self.pub_markers = self.create_publisher(
            MarkerArray, OUTPUT_TOPIC_MARKERS, QOS_DEPTH)

        self.get_logger().info(
            f"[topic] sub: {INPUT_TOPIC} -> pub: {OUTPUT_TOPIC_VISION}, {OUTPUT_TOPIC_MARKERS}")

        return config_path, ckpt_path

    def init_centerpoint(self, config_path, ckpt_path):
        self.get_logger().info('----------------- Det3D CenterPoint (Waymo) -------------------------')
        self.cfg = Config.fromfile(config_path)
        self.model = build_detector(self.cfg.model, train_cfg=None, test_cfg=self.cfg.test_cfg)
        checkpoint = load_checkpoint(self.model, ckpt_path, map_location="cpu")

        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

        self.voxel_cfg = self.cfg.voxel_generator
        self.get_logger().info(f"Voxel range: {self.voxel_cfg.range}")
        self.get_logger().info(f"Voxel size: {self.voxel_cfg.voxel_size}")

    def start_inference_thread(self):
        self._start_worker()
        threading.Thread(target=self._watchdog, daemon=True).start()

    def _start_worker(self):
        self._worker = threading.Thread(target=self.inference_loop, daemon=True)
        self._worker.start()

    def _watchdog(self):
        while rclpy.ok():
            if not (hasattr(self, "_worker") and self._worker.is_alive()):
                self.get_logger().warn("[watchdog] inference thread died. Restarting...")
                self._start_worker()
            time.sleep(1.0)

    def lidar_callback(self, msg):
        with self.lock:
            self.latest_msg = msg

    def inference_loop(self):
        rate = SafeRate(self, INFERENCE_RATE)
        while rclpy.ok():
            msg = None
            with self.lock:
                if self.latest_msg is not None:
                    msg = self.latest_msg
                    self.latest_msg = None

            if msg is not None:
                try:
                    self.process_lidar(msg)
                except Exception as e:
                    self.get_logger().error(f"[inference_loop] error: {e}")
                    import traceback
                    self.get_logger().error(traceback.format_exc())
            rate.sleep()

    def process_lidar(self, msg):
        t_start = time.time()

        # 1) Read points
        field_names = [f.name for f in msg.fields]
        has_intensity = ('intensity' in field_names)

        try:
            if has_intensity:
                pcl_iter = pc2.read_points(msg, skip_nans=True, field_names=["x", "y", "z", "intensity"])
                num_fields = 4
            else:
                pcl_iter = pc2.read_points(msg, skip_nans=True, field_names=["x", "y", "z"])
                num_fields = 3

            pcl_list = list(pcl_iter)
            if len(pcl_list) > 0:
                # Convert the structured array into a plain array
                if has_intensity:
                    np_raw = np.array([[p[0], p[1], p[2], p[3]] for p in pcl_list], dtype=np.float32)
                else:
                    np_raw = np.array([[p[0], p[1], p[2]] for p in pcl_list], dtype=np.float32)
            else:
                np_raw = np.array([], dtype=np.float32).reshape(0, num_fields)
        except Exception as e:
            self.get_logger().warn(f"[read_points] error: {e}")
            return

        # DEBUG: raw point cloud size
        if DEBUG_ENABLE:
            self.get_logger().info(f"[DEBUG] Raw points: {len(np_raw)}")

        # 2) Preprocess
        points = sanitize_points(
            np_raw,
            has_intensity=has_intensity,
            max_range=PC_MAX_RANGE,
            min_points=PC_MIN_POINTS,
            z_min=PC_Z_MIN,
            z_max=PC_Z_MAX,
            r_min=PC_R_MIN,
            intensity_max=INTENSITY_MAX
        )
        if points is None:
            if DEBUG_ENABLE:
                self.get_logger().warn(f"[DEBUG] sanitize_points failed - Raw: {len(np_raw)} points")
            self.get_logger().warn("[sanitize_points] no valid points")
            return

        # DEBUG: point count after preprocessing
        if DEBUG_ENABLE:
            self.get_logger().info(f"[DEBUG] After sanitize: {len(points)} points ({len(points)/len(np_raw)*100:.1f}% kept)")

        # 3) Run voxelization directly
        from det3d.core.input.voxel_generator import VoxelGenerator

        if not hasattr(self, 'voxel_generator'):
            self.voxel_generator = VoxelGenerator(
                voxel_size=self.voxel_cfg.voxel_size,
                point_cloud_range=self.voxel_cfg.range,
                max_num_points=self.voxel_cfg.max_points_in_voxel,
                max_voxels=self.voxel_cfg.max_voxel_num[1]
            )

        try:
            voxels, coords, num_points_per_voxel = self.voxel_generator.generate(points)
        except Exception as e:
            self.get_logger().error(f"[voxelize] error: {e}")
            return

        batch_idx = np.zeros((coords.shape[0], 1), dtype=coords.dtype)
        coords = np.concatenate([batch_idx, coords], axis=1)

        # 4) Compute the grid size
        voxel_size = np.array(self.voxel_cfg.voxel_size)
        pc_range = np.array(self.voxel_cfg.range)

        nx = int(np.round((pc_range[3] - pc_range[0]) / voxel_size[0]))
        ny = int(np.round((pc_range[4] - pc_range[1]) / voxel_size[1]))
        nz = int(np.round((pc_range[5] - pc_range[2]) / voxel_size[2]))

        # 5) Build the model input
        example = {
            'points': [points],
            'voxels': voxels,
            'coordinates': coords,
            'num_points': num_points_per_voxel,
            'num_voxels': np.array([voxels.shape[0]]),
            'shape': [[nx, ny, nz]],
            'input_shape': [nx, ny, nz],
            'metadata': [{
                'token': 'ros_frame',
                'num_point_features': 5
            }]
        }

        # 6) Convert to tensors and move to GPU
        example = self.example_convert_to_torch(example)

        # 7) Inference
        with torch.no_grad():
            try:
                outputs = self.model(example, return_loss=False)
            except Exception as e:
                self.get_logger().error(f"[model] error: {e}")
                import traceback
                self.get_logger().error(traceback.format_exc())
                return

        # 8) Post-process the results
        if len(outputs) == 0 or len(outputs[0]) == 0:
            self.get_logger().info("[inference] no detections")
            return

        pred = outputs[0]
        if 'box3d_lidar' not in pred:
            return

        boxes = pred['box3d_lidar'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['label_preds'].cpu().numpy()

        # DEBUG: raw model output
        if DEBUG_ENABLE:
            score_dist = f"min={scores.min():.3f}, max={scores.max():.3f}, mean={scores.mean():.3f}" if len(scores) > 0 else "N/A"
            self.get_logger().info(f"[DEBUG] Raw detections: {len(boxes)} | Score dist: {score_dist}")
            if len(boxes) > 0:
                class_counts = {int(l): int(np.sum(labels == l)) for l in np.unique(labels)}
                self.get_logger().info(f"[DEBUG] Raw class counts: {class_counts}")

        # 8.5) Apply NMS to drop duplicate detections
        if NMS_ENABLE and len(boxes) > 0:
            keep_indices = self.apply_nms_3d(boxes, scores, labels)
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            labels = labels[keep_indices]

            # DEBUG: results after NMS
            if DEBUG_ENABLE:
                self.get_logger().info(f"[DEBUG] After NMS: {len(boxes)} detections (removed {len(pred['box3d_lidar']) - len(boxes)})")

        # 9) Publish a vision_msgs Detection3DArray
        detections_vision = Detection3DArray()
        detections_vision.header.frame_id = msg.header.frame_id
        detections_vision.header.stamp = msg.header.stamp

        # 10) Build the MarkerArray
        markers = MarkerArray()

        # Delete marker that clears the previous frame
        delete_marker = Marker()
        delete_marker.header.frame_id = msg.header.frame_id
        delete_marker.header.stamp = msg.header.stamp
        delete_marker.action = Marker.DELETEALL
        markers.markers.append(delete_marker)

        # DEBUG: track filtering statistics
        if DEBUG_ENABLE:
            filter_stats = {
                'total': len(boxes),
                'score_filtered': 0,
                'nan_filtered': 0,
                'size_filtered': 0,
                'kept': 0
            }

        num_kept = 0
        for i in range(len(boxes)):
            if scores[i] < SCORE_THRESHOLD:
                if DEBUG_ENABLE:
                    filter_stats['score_filtered'] += 1
                continue

            x, y, z, dx, dy, dz, heading = boxes[i]

            if not np.isfinite([x, y, z, dx, dy, dz, heading]).all():
                if DEBUG_ENABLE:
                    filter_stats['nan_filtered'] += 1
                continue
            if dx <= BBOX_MIN_SIZE or dy <= BBOX_MIN_SIZE or dz <= BBOX_MIN_SIZE:
                if DEBUG_ENABLE:
                    filter_stats['size_filtered'] += 1
                continue

            q = Quaternion(axis=(0, 0, 1), radians=float(heading)).normalised

            # Build the vision_msgs Detection3D
            detection = Detection3D()
            detection.header.frame_id = msg.header.frame_id
            detection.header.stamp = msg.header.stamp

            # Fill in the BoundingBox3D
            detection.bbox.center.position.x = float(x)
            detection.bbox.center.position.y = float(y)
            detection.bbox.center.position.z = float(z)
            detection.bbox.center.orientation.x = float(q.x)
            detection.bbox.center.orientation.y = float(q.y)
            detection.bbox.center.orientation.z = float(q.z)
            detection.bbox.center.orientation.w = float(q.w)

            detection.bbox.size.x = float(dx)
            detection.bbox.size.y = float(dy)
            detection.bbox.size.z = float(dz)

            # Fill in the ObjectHypothesis
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(int(labels[i]))
            hypothesis.hypothesis.score = float(scores[i])
            detection.results.append(hypothesis)

            detections_vision.detections.append(detection)

            # MarkerArray: build the cube marker
            marker = Marker()
            marker.header.frame_id = msg.header.frame_id
            marker.header.stamp = msg.header.stamp
            marker.ns = "detections"
            marker.id = num_kept
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.position.z = float(z)
            marker.pose.orientation.x = float(q.x)
            marker.pose.orientation.y = float(q.y)
            marker.pose.orientation.z = float(q.z)
            marker.pose.orientation.w = float(q.w)

            marker.scale.x = float(dx)
            marker.scale.y = float(dy)
            marker.scale.z = float(dz)

            # Per-class color (Waymo: 0=Vehicle, 1=Pedestrian, 2=Cyclist)
            label_int = int(labels[i])
            if label_int == 0:  # Vehicle
                marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.3)
            elif label_int == 1:  # Pedestrian
                marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.3)
            elif label_int == 2:  # Cyclist
                marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.3)
            else:
                marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.3)

            marker.lifetime.sec = 1
            marker.lifetime.nanosec = 0

            markers.markers.append(marker)
            num_kept += 1
            if DEBUG_ENABLE:
                filter_stats['kept'] += 1

        # DEBUG: report filtering statistics
        if DEBUG_ENABLE:
            self.get_logger().info(
                f"[DEBUG] Filtering: Total={filter_stats['total']} | "
                f"Score<{SCORE_THRESHOLD}: {filter_stats['score_filtered']} | "
                f"NaN: {filter_stats['nan_filtered']} | "
                f"Size: {filter_stats['size_filtered']} | "
                f"Kept: {filter_stats['kept']}"
            )
            # Final per-class statistics
            if num_kept > 0:
                kept_labels = labels[[i for i in range(len(boxes)) if scores[i] >= SCORE_THRESHOLD and
                                     np.isfinite(boxes[i]).all() and
                                     boxes[i][3] > BBOX_MIN_SIZE and boxes[i][4] > BBOX_MIN_SIZE and boxes[i][5] > BBOX_MIN_SIZE]]
                class_counts_final = {int(l): int(np.sum(kept_labels == l)) for l in np.unique(kept_labels)}
                self.get_logger().info(f"[DEBUG] Final class counts: {class_counts_final}")

        t_elapsed = time.time() - t_start
        fps = 1.0 / t_elapsed if t_elapsed > 0 else 0
        self.get_logger().info(f"[publish] boxes: {num_kept} | time: {t_elapsed*1000:.1f}ms | FPS: {fps:.1f}")
        self.pub_vision.publish(detections_vision)
        self.pub_markers.publish(markers)

    def apply_nms_3d(self, boxes, scores, labels):
        """
        Apply 3D NMS (Non-Maximum Suppression).
        - Drops duplicate detections with a high IoU within the same class
        - Keeps the higher-scoring box

        Args:
            boxes: (N, 7) array [x, y, z, dx, dy, dz, heading]
            scores: (N,) array
            labels: (N,) array

        Returns:
            keep_indices: indices of the boxes to keep
        """
        if len(boxes) == 0:
            return np.array([], dtype=np.int32)

        keep_indices = []

        # Apply NMS per class
        unique_labels = np.unique(labels)
        for label in unique_labels:
            # Take only the boxes of this class
            class_mask = labels == label
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]
            class_indices = np.where(class_mask)[0]

            # Sort by score, descending
            sorted_indices = np.argsort(-class_scores)

            keep_mask = np.ones(len(class_boxes), dtype=bool)

            for i in range(len(sorted_indices)):
                if not keep_mask[sorted_indices[i]]:
                    continue

                box_i = class_boxes[sorted_indices[i]]

                # Compute the IoU against the remaining boxes
                for j in range(i + 1, len(sorted_indices)):
                    if not keep_mask[sorted_indices[j]]:
                        continue

                    box_j = class_boxes[sorted_indices[j]]

                    # Compute the BEV (bird's eye view) IoU
                    iou = self.calculate_bev_iou(box_i, box_j)

                    # Drop the box when the IoU exceeds the threshold
                    if iou > NMS_IOU_THRESHOLD:
                        keep_mask[sorted_indices[j]] = False

            # Record the indices to keep
            keep_indices.extend(class_indices[sorted_indices[keep_mask[sorted_indices]]])

        return np.array(keep_indices, dtype=np.int32)

    def calculate_bev_iou(self, box1, box2):
        """
        Compute the BEV (bird's eye view) IoU.
        - Computed on the 2D (x, y) plane
        - Meant to account for box rotation

        Args:
            box1, box2: [x, y, z, dx, dy, dz, heading]

        Returns:
            iou: IoU value (0.0 ~ 1.0)
        """
        # Simple axis-aligned 2D IoU; rotation is ignored here.
        # A rotation-aware IoU would be more accurate.

        x1, y1, _, dx1, dy1, _, _ = box1
        x2, y2, _, dx2, dy2, _, _ = box2

        # Box extents
        x1_min, x1_max = x1 - dx1/2, x1 + dx1/2
        y1_min, y1_max = y1 - dy1/2, y1 + dy1/2

        x2_min, x2_max = x2 - dx2/2, x2 + dx2/2
        y2_min, y2_max = y2 - dy2/2, y2 + dy2/2

        # Intersection area
        inter_x_min = max(x1_min, x2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_min = max(y1_min, y2_min)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # Area of each box
        area1 = dx1 * dy1
        area2 = dx2 * dy2

        # Union area
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def example_convert_to_torch(self, example):
        """Convert a Det3D example into torch tensors."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        example["voxels"] = torch.tensor(
            example["voxels"], dtype=torch.float32, device=device
        )

        coords_np = example["coordinates"]

        example["coordinates"] = torch.tensor(
            coords_np, dtype=torch.int32, device=device
        )

        example["num_points"] = torch.tensor(
            example["num_points"], dtype=torch.int32, device=device
        )
        example["num_voxels"] = np.array([example["voxels"].shape[0]])

        example["coors"] = example["coordinates"]
        example["batch_size"] = 1

        ndim = example["coordinates"].shape[1]

        if "points" in example:
            example["points"] = [example["points"]]

        return example


def main(args=None):
    rclpy.init(args=args)
    node = CenterPoint_Det3D_ROS()
    node.start_inference_thread()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("Shutting down")


if __name__ == '__main__':
    main()
