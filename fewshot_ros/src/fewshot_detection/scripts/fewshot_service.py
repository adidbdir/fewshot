#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import os.path as osp
import sys
import rospy
import roslib
import cv2
import numpy as np
import torch
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from geometry_msgs.msg import Point, Quaternion, TransformStamped
from std_msgs.msg import Header
import tf2_ros

# Import the service and message types
from fewshot_detection.msg import BBox, Pose3D, FewshotDetection
from fewshot_detection.srv import (
    FewshotDetectionService, 
    FewshotDetectionServiceResponse
)

# Add the CDFSOD-benchmark directory to the path
CDFSOD_DIR = osp.join(
    osp.dirname(osp.dirname(osp.dirname(osp.dirname(
        osp.abspath(__file__)
    )))),
    "CDFSOD-benchmark"
)
sys.path.insert(0, CDFSOD_DIR)

# Import CDFSOD-benchmark demo functions
import torch
from detectron2.config import get_cfg
import detectron2.data.transforms as T
import detectron2.data.detection_utils as utils
from tools.train_net import Trainer, DetectionCheckpointer
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import matplotlib.colors
import seaborn as sns
import torchvision.ops as ops
from PIL import Image as PILImage


class FewshotDetectionNode:
    def __init__(self):
        # Initialize ROS node
        self.node_name = rospy.get_name()
        
        # Read parameters
        self.p_action_name = rospy.get_param("~action_name", "fewshot_detection")
        
        # Default model parameters
        self.p_config_file = rospy.get_param("~config_file", "configs/open-vocabulary/lvis/vitl.yaml")
        self.p_rpn_config_file = rospy.get_param("~rpn_config_file", "configs/RPN/mask_rcnn_R_50_FPN_1x.yaml")
        self.p_model_path = rospy.get_param("~model_path", "weights/trained/open-vocabulary/lvis/vitl_0069999.pth")
        self.p_device = rospy.get_param("~device", "cpu")
        
        # Detection parameters
        self.p_confidence_th = rospy.get_param("~confidence_th", 0.45)
        self.p_iou_th = rospy.get_param("~iou_th", 0.5)
        self.p_topk = rospy.get_param("~topk", 1)
        self.p_overlapping_mode = rospy.get_param("~overlapping_mode", True)
        
        # Image topics
        self.p_camera_info_topic = rospy.get_param("~camera_info_topic", "/camera/rgb/camera_info")
        self.p_rgb_topic = rospy.get_param("~rgb_topic", "/camera/rgb/image_raw/compressed")
        self.p_use_depth = rospy.get_param("~use_depth", False)
        self.p_depth_topic = rospy.get_param("~depth_topic", "/camera/depth_registered/image_raw/compressedDepth")
        
        # Service parameters
        self.p_use_latest_image = rospy.get_param("~use_latest_image", False)
        self.p_max_distance = rospy.get_param("~max_distance", -1)
        self.p_category_space_default = rospy.get_param("~category_space", "demo/ycb_prototypes.pth")
        
        # Save default values
        self.p_confidence_th_default = self.p_confidence_th
        self.p_iou_th_default = self.p_iou_th
        self.p_max_distance_default = self.p_max_distance
        
        # Initialize variables
        self.cv_bgr = None
        self.msg_rgb = None
        self.cv_depth = None
        self.msg_depth = None
        self.camera_info = None
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Initialize TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # Setup subscribers
        self.setup_subscribers()
        
        # Setup publishers
        self.setup_publishers()
        
        # Load model
        self.setup_model()
        
        # Register service
        self.srv_detection = rospy.Service(
            f"{self.p_action_name}/service", 
            FewshotDetectionService, 
            self.run
        )
        rospy.loginfo(f"FewshotDetection Service is ready.")
    
    def setup_subscribers(self):
        # Subscribe to camera info
        self.sub_camera_info = rospy.Subscriber(
            self.p_camera_info_topic, 
            CameraInfo, 
            self.callback_camera_info
        )
        
        # Subscribe to RGB image
        self.sub_rgb = rospy.Subscriber(
            self.p_rgb_topic, 
            CompressedImage, 
            self.callback_rgb
        )
        
        # Subscribe to depth image if depth is enabled
        if self.p_use_depth:
            self.sub_depth = rospy.Subscriber(
                self.p_depth_topic, 
                CompressedImage, 
                self.callback_depth
            )
    
    def setup_publishers(self):
        # Publisher for result image
        self.pub_result_image = rospy.Publisher(
            f"{self.p_action_name}/result_image/compressed", 
            CompressedImage, 
            queue_size=1
        )
    
    def setup_model(self):
        # Initialize model
        rospy.loginfo("Loading fewshot detection model...")
        
        # Create config 
        self.cfg = get_cfg()
        
        # Load RPN config
        self.cfg.merge_from_file(osp.join(CDFSOD_DIR, self.p_rpn_config_file))
        
        # Load main config
        self.cfg.merge_from_file(osp.join(CDFSOD_DIR, self.p_config_file))
        
        # Set device
        self.device = torch.device(self.p_device)
        
        # Load model
        self.trainer = Trainer.from_config(self.cfg)
        self.trainer.checkpointer = DetectionCheckpointer(
            self.trainer.model, 
            save_dir=self.cfg.OUTPUT_DIR
        )
        self.trainer.checkpointer.load(osp.join(CDFSOD_DIR, self.p_model_path))
        self.trainer.model.eval()
        self.trainer.model = self.trainer.model.to(self.device)
        
        rospy.loginfo("Model loaded successfully.")
    
    def load_categories(self, category_space):
        # Load the category space
        try:
            if osp.isfile(category_space):
                category_path = category_space
            else:
                category_path = osp.join(CDFSOD_DIR, category_space)
                
            rospy.loginfo(f"Loading category space from: {category_path}")
            categories = torch.load(category_path)
            return categories
        except Exception as e:
            rospy.logerr(f"Error loading category space: {e}")
            return None
    
    def set_params(self, req):
        # Set parameters from request
        if req.confidence_th <= 0:
            self.p_confidence_th = self.p_confidence_th_default
        else:
            self.p_confidence_th = req.confidence_th
            
        if req.iou_th <= 0:
            self.p_iou_th = self.p_iou_th_default
        else:
            self.p_iou_th = req.iou_th
            
        self.p_use_latest_image = req.use_latest_image
        
        if req.max_distance != 0:
            self.p_max_distance = req.max_distance
        else:
            self.p_max_distance = self.p_max_distance_default
            
        if req.category_space:
            self.p_category_space = req.category_space
        else:
            self.p_category_space = self.p_category_space_default
            
        if req.topk > 0:
            self.p_topk = req.topk
        
        rospy.loginfo(f"Parameters set: confidence_th={self.p_confidence_th}, iou_th={self.p_iou_th}, topk={self.p_topk}")
    
    def callback_camera_info(self, msg):
        self.camera_info = msg
    
    def callback_rgb(self, msg):
        try:
            self.msg_rgb = msg
            self.cv_bgr = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"Error converting RGB image: {e}")
    
    def callback_depth(self, msg):
        try:
            self.msg_depth = msg
            self.cv_depth = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            rospy.logerr(f"Error converting depth image: {e}")
    
    def wait_for_message(self, attr_name, timeout=5.0):
        start_time = rospy.Time.now()
        while not rospy.is_shutdown():
            if hasattr(self, attr_name) and getattr(self, attr_name) is not None:
                return True
            
            if (rospy.Time.now() - start_time).to_sec() > timeout:
                rospy.logwarn(f"Timeout waiting for {attr_name}")
                return False
            
            rospy.sleep(0.1)
        
        return False
    
    def run_inference(self, img, category_space):
        # Preprocess the image
        height, width = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load transform
        transform_gen = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST],
            self.cfg.INPUT.MAX_SIZE_TEST
        )
        img_t = transform_gen.get_transform(img).apply_image(img)
        img_t = torch.as_tensor(img_t.astype("float32").transpose(2, 0, 1))
        
        # Prepare inputs
        inputs = {"image": img_t, "height": height, "width": width}
        
        # Load categories
        categories = self.load_categories(category_space)
        if categories is None:
            rospy.logerr("Failed to load categories")
            return None, None, None
        
        # Run inference
        with torch.no_grad():
            # Move inputs to device
            inputs = {k: torch.as_tensor(v).to(self.device) if isinstance(v, torch.Tensor) else v 
                      for k, v in inputs.items()}
            
            # Run model
            instances = self.trainer.model([inputs], [categories])[0]["instances"]
            
            # Filter by confidence
            if len(instances) > 0:
                boxes, pred_classes, scores = self.filter_boxes(instances, self.p_confidence_th)
                
                # Limit to topk detections
                if self.p_topk > 0 and len(boxes) > self.p_topk:
                    top_indices = torch.argsort(scores, descending=True)[:self.p_topk]
                    boxes = boxes[top_indices]
                    pred_classes = pred_classes[top_indices]
                    scores = scores[top_indices]
                
                return boxes, pred_classes, scores
            else:
                return None, None, None
    
    def filter_boxes(self, instances, threshold=0.0):
        indexes = instances.scores >= threshold
        if indexes.sum() > 0:
            boxes = instances.pred_boxes.tensor[indexes, :]
            pred_classes = instances.pred_classes[indexes]
            return boxes, pred_classes, instances.scores[indexes]
        else:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
    
    def get_3d_poses(self, depth_img, boxes):
        # Implementation for 3D pose estimation using depth image
        # This is a placeholder, actual implementation would depend on the camera parameters
        poses = []
        
        if depth_img is None or boxes is None:
            return poses
        
        for box in boxes:
            # Extract the box coordinates
            x1, y1, x2, y2 = box.int().cpu().numpy()
            
            # Calculate center point
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            # Get depth at center point (simple approach, could be improved)
            try:
                if 0 <= cx < depth_img.shape[1] and 0 <= cy < depth_img.shape[0]:
                    depth_value = depth_img[cy, cx]
                    
                    # Create a pose
                    pose = Pose3D()
                    pose.is_valid = True
                    pose.frame_id = "camera_depth_optical_frame"
                    
                    # Simple 3D point calculation (assuming camera parameters are known)
                    # This is a simplified calculation and should be replaced with proper camera model
                    if self.camera_info is not None:
                        fx = self.camera_info.K[0]
                        fy = self.camera_info.K[4]
                        cx_cam = self.camera_info.K[2]
                        cy_cam = self.camera_info.K[5]
                        
                        # Convert from image coords to 3D coords
                        z = depth_value
                        x = (cx - cx_cam) * z / fx
                        y = (cy - cy_cam) * z / fy
                        
                        pose.position = Point(x=x, y=y, z=z)
                    else:
                        # Default if camera info not available
                        pose.position = Point(x=0, y=0, z=depth_value)
                    
                    pose.orientation = Quaternion(x=0, y=0, z=0, w=1)
                    poses.append(pose)
                else:
                    # Out of bounds
                    pose = Pose3D()
                    pose.is_valid = False
                    poses.append(pose)
            except Exception as e:
                rospy.logerr(f"Error calculating 3D pose: {e}")
                pose = Pose3D()
                pose.is_valid = False
                poses.append(pose)
        
        return poses
    
    def visualize(self, img, boxes, classes, scores):
        # Convert to RGB for visualization
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1))
        
        # Convert class indices to labels
        labels = [f"{i}: {s:.2f}" for i, s in zip(classes.cpu().numpy(), scores.cpu().numpy())]
        
        # Generate colors
        colors = sns.color_palette("hls", len(boxes)).as_hex()
        
        # Draw boxes
        result_img = draw_bounding_boxes(
            img_tensor, 
            boxes=boxes, 
            labels=labels, 
            colors=colors, 
            width=2, 
            font="Arial.ttf", 
            font_size=20
        )
        
        # Convert back to numpy
        result_img = result_img.permute(1, 2, 0).cpu().numpy()
        
        # Convert back to BGR for OpenCV
        result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        
        return result_img
    
    def create_bbox_msgs(self, boxes, classes, scores):
        # Create BBox messages
        bbox_msgs = []
        
        if boxes is None or len(boxes) == 0:
            return bbox_msgs
        
        for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
            bbox = BBox()
            bbox.id = int(cls.item())
            bbox.name = str(cls.item())  # We would ideally map this to a class name
            bbox.score = float(score.item())
            
            # Extract coordinates
            x1, y1, x2, y2 = box.int().cpu().numpy()
            bbox.x = int(x1)
            bbox.y = int(y1)
            bbox.w = int(x2 - x1)
            bbox.h = int(y2 - y1)
            
            bbox_msgs.append(bbox)
        
        return bbox_msgs
    
    def create_object_detection_msg(self, msg_rgb, boxes, classes, scores, msg_depth=None, poses=None):
        # Create detection message
        msg = FewshotDetection()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "camera_rgb_optical_frame"
        
        # Set detection status
        msg.is_detected = boxes is not None and len(boxes) > 0
        
        # Set camera info
        if self.camera_info is not None:
            msg.camera_info = self.camera_info
        
        # Set RGB image
        if msg_rgb is not None:
            msg.rgb = msg_rgb
        
        # Set depth image
        if msg_depth is not None:
            msg.depth = msg_depth
        
        # Set bounding boxes
        if boxes is not None and len(boxes) > 0:
            msg.bbox = self.create_bbox_msgs(boxes, classes, scores)
        
        # Set poses
        if poses is not None:
            msg.pose = poses
        
        return msg
    
    def publish_tf(self, poses, classes):
        # Publish TF frames for detected objects
        if poses is None or classes is None:
            return
        
        for i, (pose, cls) in enumerate(zip(poses, classes)):
            if not pose.is_valid:
                continue
            
            # Create transform
            transform = TransformStamped()
            transform.header.stamp = rospy.Time.now()
            transform.header.frame_id = pose.frame_id
            transform.child_frame_id = f"object_{cls}_{i}"
            
            # Set translation
            transform.transform.translation.x = pose.position.x
            transform.transform.translation.y = pose.position.y
            transform.transform.translation.z = pose.position.z
            
            # Set rotation
            transform.transform.rotation = pose.orientation
            
            # Broadcast transform
            self.tf_broadcaster.sendTransform(transform)
    
    def return_no_objects(self, bgr=None):
        # Return empty detection result
        if bgr is not None:
            # Publish result image if available
            try:
                msg_result = self.bridge.cv2_to_compressed_imgmsg(bgr)
                self.pub_result_image.publish(msg_result)
            except CvBridgeError as e:
                rospy.logerr(f"Error converting result image: {e}")
        
        # Create empty detection message
        msg = FewshotDetection()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.is_detected = False
        
        # Add camera info if available
        if self.camera_info is not None:
            msg.camera_info = self.camera_info
        
        return FewshotDetectionServiceResponse(detections=msg)
    
    def run(self, req):
        try:
            # Set parameters from request
            self.set_params(req)
            
            # Wait for RGB image
            while not rospy.is_shutdown():
                if not hasattr(self, "cv_bgr") or self.cv_bgr is None:
                    rospy.loginfo("Waiting for RGB image...")
                    if not self.wait_for_message("cv_bgr"):
                        return self.return_no_objects()
                    continue
                
                if self.p_use_latest_image:
                    if not self.wait_for_message("msg_rgb"):
                        return self.return_no_objects()
                
                cv_bgr = self.cv_bgr
                msg_rgb = self.msg_rgb
                break
            
            # Run inference
            try:
                boxes, classes, scores = self.run_inference(cv_bgr, self.p_category_space)
                if boxes is None or len(boxes) == 0:
                    rospy.logwarn("No objects detected.")
                    return self.return_no_objects(cv_bgr)
            except Exception as e:
                rospy.logerr(f"Inference error: {e}")
                return self.return_no_objects(cv_bgr)
            
            # Process depth if available
            poses = None
            if self.p_use_depth:
                if not hasattr(self, "cv_depth") or self.cv_depth is None:
                    rospy.logwarn("Depth image not available")
                else:
                    cv_depth = self.cv_depth
                    msg_depth = self.msg_depth
                    
                    # Get 3D poses
                    poses = self.get_3d_poses(cv_depth, boxes)
                    
                    # Publish TF frames
                    self.publish_tf(poses, classes)
            else:
                msg_depth = None
            
            # Visualize results
            cv_result = self.visualize(cv_bgr, boxes, classes, scores)
            
            # Publish result image
            try:
                msg_result = self.bridge.cv2_to_compressed_imgmsg(cv_result)
                self.pub_result_image.publish(msg_result)
            except CvBridgeError as e:
                rospy.logerr(f"Error converting result image: {e}")
            
            # Create response
            result = FewshotDetectionServiceResponse()
            result.detections = self.create_object_detection_msg(
                msg_rgb, boxes, classes, scores, msg_depth, poses
            )
            
            return result
            
        except Exception as e:
            rospy.logerr(f"Service error: {e}")
            return self.return_no_objects()
    
    def shutdown(self):
        # Shutdown node
        rospy.loginfo("Shutting down fewshot detection service")


def main():
    # Initialize ROS node
    rospy.init_node('fewshot_detection_service')
    
    # Get loop rate parameter
    p_loop_rate = rospy.get_param("~loop_rate", 30)
    loop_wait = rospy.Rate(p_loop_rate)
    
    # Create and initialize the node
    node = FewshotDetectionNode()
    
    # Register shutdown hook
    rospy.on_shutdown(node.shutdown)
    
    # Main loop
    rospy.loginfo("Fewshot detection service ready")
    while not rospy.is_shutdown():
        loop_wait.sleep()


if __name__ == "__main__":
    main() 