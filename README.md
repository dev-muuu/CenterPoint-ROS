# centerpoint_ws

ROS 2 wrapper for the CenterPoint 3D object detector (Det3D, Waymo model).

- Subscribes: `/pointcloud/disturbance` (`sensor_msgs/PointCloud2`)
- Publishes: `/detections_vision` (`vision_msgs/Detection3DArray`),
  `/detections_markers` (`visualization_msgs/MarkerArray`)

Topics, QoS and detection thresholds are configured at the top of
`src/centerpoint_ros/scripts/ros.py`.

## Build

```bash
cd /home/centerpoint_ws
colcon build --symlink-install
```

## Run

```bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/home/centerpoint_ws/fastdds.xml
export ROS_DOMAIN_ID=0

source install/setup.bash
ros2 launch centerpoint_ros launch.py
```

## Paths

If cloned elsewhere:

```bash
export CENTERPOINT_ROS_ROOT=/path/to/centerpoint_ws/src/centerpoint_ros
```

Rebuild the CUDA extensions if your PyTorch/CUDA version differs:

```bash
cd src/centerpoint_ros/det3d/ops/iou3d_nms && python3 setup.py build_ext --inplace
cd ../dcn && python3 setup.py build_ext --inplace
```
