#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import sys
from fewshot_detection.srv import FewshotDetectionService
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

def test_fewshot_detection_client():
    rospy.init_node('fewshot_detection_client')
    rospy.loginfo("Waiting for fewshot_detection service...")
    
    # Wait for service to become available
    rospy.wait_for_service('fewshot_detection/service')
    
    try:
        # Create a service proxy
        fewshot_detection = rospy.ServiceProxy('fewshot_detection/service', FewshotDetectionService)
        
        # Create a request
        confidence_th = 0.5
        iou_th = 0.5
        use_latest_image = True
        max_distance = -1.0
        category_space = "demo/ycb_prototypes.pth"
        topk = 5
        
        rospy.loginfo("Calling fewshot_detection service with: ")
        rospy.loginfo(f"  confidence_th: {confidence_th}")
        rospy.loginfo(f"  iou_th: {iou_th}")
        rospy.loginfo(f"  use_latest_image: {use_latest_image}")
        rospy.loginfo(f"  max_distance: {max_distance}")
        rospy.loginfo(f"  category_space: {category_space}")
        rospy.loginfo(f"  topk: {topk}")
        
        # Call the service
        response = fewshot_detection(
            confidence_th=confidence_th,
            iou_th=iou_th,
            use_latest_image=use_latest_image,
            max_distance=max_distance,
            category_space=category_space,
            topk=topk
        )
        
        # Process the response
        detections = response.detections
        rospy.loginfo(f"Got response, is_detected: {detections.is_detected}")
        
        if detections.is_detected:
            rospy.loginfo(f"Number of detections: {len(detections.bbox)}")
            
            # Display detection information
            for i, bbox in enumerate(detections.bbox):
                rospy.loginfo(f"  Detection {i+1}:")
                rospy.loginfo(f"    ID: {bbox.id}")
                rospy.loginfo(f"    Name: {bbox.name}")
                rospy.loginfo(f"    Score: {bbox.score}")
                rospy.loginfo(f"    Position: ({bbox.x}, {bbox.y}), ({bbox.w}, {bbox.h})")
                
                # If poses are available
                if i < len(detections.pose):
                    pose = detections.pose[i]
                    if pose.is_valid:
                        rospy.loginfo(f"    3D Position: ({pose.position.x}, {pose.position.y}, {pose.position.z})")
            
            # Display result image if available
            if detections.rgb is not None:
                try:
                    bridge = CvBridge()
                    cv_image = bridge.compressed_imgmsg_to_cv2(detections.rgb, desired_encoding="bgr8")
                    cv2.imshow("Detection Result", cv_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                except CvBridgeError as e:
                    rospy.logerr(f"Error converting image: {e}")
        else:
            rospy.loginfo("No objects detected.")
        
        return detections.is_detected
        
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        return False

if __name__ == "__main__":
    test_fewshot_detection_client() 