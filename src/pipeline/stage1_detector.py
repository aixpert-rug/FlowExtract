"""
Stage 1: Element Detection

YOLO-based detection for 6 TOCAP element types.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
from ultralytics import YOLO
import cv2
import numpy as np

from ..utils.io_utils import save_json, ensure_dir
from ..utils.visualization import draw_bounding_boxes


class YOLODetector:
    """YOLO-based element detector using ultralytics YOLO."""
    
    def __init__(self, config: Dict):
        """
        Initialize YOLO detector.
        
        Args:
            config: Configuration dictionary for YOLO detection
        """
        self.config = config
        self.model_path = config.get('model_path')
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.iou_threshold = config.get('iou_threshold', 0.4)
        self.model = None
        
        # Load model if path exists
        if self.model_path and os.path.exists(self.model_path):
            self._load_model()
    
    def _load_model(self):
        """Load YOLO model using ultralytics."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            print(f"Loaded YOLO model from: {self.model_path}")
        except Exception as e:
            print(f"Warning: Could not load YOLO model: {e}")
            self.model = None
    
    def detect(self, image_path: str) -> List[Dict]:
        """
        Detect elements using YOLO model.
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of detected elements with bounding boxes
        """
        if self.model is None:
            raise ValueError(
                "YOLO model not loaded. Please train a YOLO model and specify the path "
                "in the configuration file (detection.yolo.model_path)."
            )
        
        # Run inference
        results = self.model.predict(
            image_path,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # Extract detections
        elements = []
        
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for idx, box in enumerate(boxes):
                # Get bounding box coordinates (xyxy format)
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                
                # Convert to xywh format
                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)
                
                # Get confidence and class
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                elements.append({
                    'id': idx,
                    'bbox': [x, y, w, h],
                    'confidence': confidence,
                    'class_id': class_id,
                    'area': w * h,
                    'aspect_ratio': w / h if h > 0 else 0
                })
        
        return elements


class ElementDetector:
    """Main element detection interface using YOLO."""
    
    def __init__(self, config: Dict):
        """
        Initialize element detector.
        
        Args:
            config: Full pipeline configuration
        """
        self.config = config
        detection_config = config.get('detection', {})
        
        # Initialize YOLO detector
        self.detector = YOLODetector(detection_config.get('yolo', {}))
    
    def process_image(
        self,
        image_path: str,
        output_dir: str,
        save_visualization: bool = True
    ) -> Dict:
        """
        Process single image and detect elements.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save outputs
            save_visualization: Whether to save visualization image
            
        Returns:
            Detection results dictionary
        """
        # Detect elements
        elements = self.detector.detect(image_path)
        
        # Prepare output
        image_name = Path(image_path).stem
        result = {
            'image_path': image_path,
            'image_name': image_name,
            'num_elements': len(elements),
            'elements': elements
        }
        
        # Save JSON
        ensure_dir(output_dir)
        json_path = os.path.join(output_dir, f"{image_name}_detection.json")
        save_json(result, json_path)
        
        # Save visualization
        if save_visualization:
            image = cv2.imread(image_path)
            vis_image = draw_bounding_boxes(image, elements)
            vis_path = os.path.join(output_dir, f"{image_name}_detection_vis.png")
            cv2.imwrite(vis_path, vis_image)
        
        return result
    
    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        save_visualization: bool = True
    ) -> List[Dict]:
        """
        Process all images in directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save outputs
            save_visualization: Whether to save visualization images
            
        Returns:
            List of detection results for all images
        """
        from ..utils.io_utils import get_image_paths
        
        image_paths = get_image_paths(input_dir)
        results = []
        
        print(f"Processing {len(image_paths)} images...")
        for image_path in image_paths:
            try:
                result = self.process_image(image_path, output_dir, save_visualization)
                results.append(result)
                print(f"✓ Processed {Path(image_path).name}: {result['num_elements']} elements")
            except Exception as e:
                print(f"✗ Error processing {image_path}: {e}")
        
        return results
