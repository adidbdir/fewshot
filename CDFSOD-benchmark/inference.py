from detectron2.engine.defaults import DefaultPredictor
from detectron2.config import get_cfg
import cv2

# Set up configuration
cfg = get_cfg()
# Load your specific config file or set configuration parameters
cfg.MODEL.WEIGHTS = "tam_object_fewshot/third_party/CDFSOD-benchmark/weights/vitl_0089999.pth"  # Path to the trained model
cfg.MODEL.DEVICE = "cuda"  # or "cpu" if you don't have a GPU

# Create predictor
predictor = DefaultPredictor(cfg)

# Load image
image = cv2.imread("tam_object_fewshot/third_party/CDFSOD-benchmark/demo/input/ycb.jpg")

# Run inference
outputs = predictor(image)
print(outputs)
# cv2.imwrite("tam_object_fewshot/third_party/CDFSOD-benchmark/demo/output/test.jpg", outputs["instances"].pred_masks.cpu().numpy())
# Process outputs (e.g., visualize or save results)
# outputs contains predictions with fields like "instances" that has
# pred_boxes, scores, pred_classes, etc.
