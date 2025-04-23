# Fewshot Detection ROS Package

This ROS package provides a service interface for few-shot object detection using the CDFSOD-benchmark model.

## Overview

The package provides a ROS service that:
- Takes an input RGB image (and optionally a depth image)
- Uses a pre-trained model to detect objects in the image
- Returns detection results (bounding boxes, class IDs, and optionally 3D poses)
- Visualizes the results and publishes them as images

## Prerequisites

- ROS (tested on ROS Noetic)
- Python 3.6+
- PyTorch 1.7+
- CUDA (for GPU acceleration)
- Detectron2
- CDFSOD-benchmark model and weights

## Installation

1. Clone this repository into your ROS workspace:
   ```
   cd <your_workspace>/src
   git clone https://github.com/your_username/fewshot_ros.git
   ```

2. Build the package:
   ```
   cd <your_workspace>
   catkin_make
   ```

3. Source your workspace:
   ```
   source devel/setup.bash
   ```

## Usage

### Start the service

```bash
roslaunch fewshot_detection fewshot_detection_service.launch
```

### Service Parameters

You can customize the service by modifying the launch file parameters:

- `device`: Device to run the model on (default: "cpu", use "cuda:0" for GPU)
- `confidence_th`: Confidence threshold for detections (default: 0.45)
- `iou_th`: IOU threshold for non-maximum suppression (default: 0.5)
- `topk`: Maximum number of detections to return (default: 1)
- `category_space`: Path to the categories file (default: "demo/ycb_prototypes.pth")
- `use_depth`: Whether to use depth information (default: false)
- `rgb_topic`: Topic for RGB images (default: "/camera/rgb/image_raw/compressed")
- `depth_topic`: Topic for depth images (default: "/camera/depth_registered/image_raw/compressedDepth")

### Test the service

```bash
rosrun fewshot_detection test_client.py
```

## Service Interface

### Request
- `confidence_th` (float32): Confidence threshold for detections
- `iou_th` (float32): IOU threshold for non-maximum suppression
- `use_latest_image` (bool): Whether to wait for the latest image
- `max_distance` (float32): Maximum distance for depth filtering
- `category_space` (string): Path to the category space file
- `topk` (int32): Maximum number of detections to return

### Response
- `detections` (FewshotDetection): Detection results
  - `header` (Header): Standard ROS header
  - `is_detected` (bool): Whether any objects were detected
  - `camera_info` (CameraInfo): Camera information
  - `rgb` (CompressedImage): Original RGB image
  - `depth` (CompressedImage): Original depth image
  - `bbox` (BBox[]): Bounding boxes for detected objects
  - `segments` (CompressedImage[]): Segmentation masks
  - `pose` (Pose3D[]): 3D poses of detected objects

## License

MIT

## Author

Your Name 