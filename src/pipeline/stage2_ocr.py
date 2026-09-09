"""
Stage 2: OCR Module

Extracts text content from detected elements using PaddleOCR or EasyOCR.
Preserves line breaks and bullet point structure.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional
import os
from pathlib import Path

from ..utils.io_utils import load_json, save_json, ensure_dir


class OCREngine:
    """OCR engine for text extraction from element images."""
    
    def __init__(self, config: Dict):
        """
        Initialize OCR engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        ocr_config = config.get('ocr', {})
        
        self.engine_name = ocr_config.get('engine', 'paddleocr')
        self.language = ocr_config.get('language', 'nl')
        self.use_gpu = ocr_config.get('use_gpu', True)
        self.confidence_threshold = ocr_config.get('confidence_threshold', 0.5)
        self.preserve_line_breaks = ocr_config.get('preserve_line_breaks', True)
        
        self.ocr = self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize OCR engine."""
        if self.engine_name == 'paddleocr':
            from paddleocr import PaddleOCR
            
            # Use minimal parameters for compatibility with newer PaddleOCR versions
            ocr = PaddleOCR(
                use_angle_cls=True,
                lang=self.language
            )
            return ocr
        
        elif self.engine_name == 'easyocr':
            import easyocr
            
            # Initialize with optimized parameters for industrial diagrams
            ocr = easyocr.Reader(
                [self.language],
                gpu=self.use_gpu,
                verbose=False
            )
            return ocr
        
        else:
            raise ValueError(f"Unknown OCR engine: {self.engine_name}")
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed image (grayscale)
        """
        # 1. Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 2. Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 3. Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
        
        # 4. Binarize using Otsu's thresholding
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 5. Add padding (OCR works better with margins)
        padded = cv2.copyMakeBorder(binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        
        # 6. Resize if too small (minimum height for good OCR)
        h, w = padded.shape
        if h < 32:
            scale = 32 / h
            padded = cv2.resize(padded, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        return padded
    
    def extract_text(self, image: np.ndarray) -> Dict:
        """
        Extract text from image with preprocessing.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary with extracted text and metadata
        """
        # Preprocess image for better OCR
        preprocessed = self._preprocess_image(image)
        
        if self.engine_name == 'paddleocr':
            return self._extract_paddleocr(preprocessed)
        elif self.engine_name == 'easyocr':
            return self._extract_easyocr(preprocessed)
    
    def _extract_paddleocr(self, image: np.ndarray) -> Dict:
        """Extract text using PaddleOCR."""
        # Newer PaddleOCR versions don't support cls parameter
        result = self.ocr.ocr(image)
        
        if result is None or len(result) == 0 or result[0] is None:
            return {
                'text': '',
                'lines': [],
                'confidence': 0.0
            }
        
        # Parse results - newer API returns different format
        lines = []
        confidences = []
        
        for line in result[0]:
            if line and len(line) >= 2:
                # line format: [[bbox], (text, confidence)]
                text_info = line[1]
                if isinstance(text_info, (tuple, list)) and len(text_info) >= 2:
                    text = text_info[0]
                    confidence = text_info[1]
                elif isinstance(text_info, str):
                    # Fallback if format is just string
                    text = text_info
                    confidence = 1.0
                else:
                    continue
                    
                if confidence >= self.confidence_threshold:
                    lines.append(text)
                    confidences.append(confidence)
        
        # Combine lines
        if self.preserve_line_breaks:
            full_text = '\n'.join(lines)
        else:
            full_text = ' '.join(lines)
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            'text': full_text,
            'lines': lines,
            'confidence': float(avg_confidence)
        }
    
    def _extract_easyocr(self, image: np.ndarray) -> Dict:
        """Extract text using EasyOCR with optimized parameters."""
        # Use optimized parameters for better accuracy
        result = self.ocr.readtext(
            image,
            detail=1,  # Return bounding boxes and confidence
            paragraph=True,  # Group text into paragraphs
            min_size=10,  # Minimum text size in pixels
            text_threshold=0.7,  # Higher confidence threshold
            low_text=0.4,  # Lower bound for text detection
            link_threshold=0.4,  # Link threshold for text grouping
            canvas_size=2560,  # Larger canvas for better quality
            mag_ratio=1.5  # Magnification ratio
        )
        
        if not result:
            return {
                'text': '',
                'lines': [],
                'confidence': 0.0
            }
        
        lines = []
        confidences = []
        
        for detection in result:
            # Handle both formats: [bbox, text, confidence] or [bbox, text]
            if len(detection) >= 3:
                text = detection[1]
                confidence = detection[2]
            elif len(detection) == 2:
                text = detection[1]
                confidence = 1.0  # Default confidence if not provided
            else:
                continue  # Skip malformed detections
            
            if confidence >= self.confidence_threshold:
                lines.append(text)
                confidences.append(confidence)
        
        # Smart text joining - use spaces instead of newlines
        full_text = ' '.join(lines)
        # Collapse multiple spaces
        import re
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            'text': full_text,
            'lines': lines,
            'confidence': float(avg_confidence)
        }
    
    def process_elements(
        self,
        image_path: str,
        detection_json_path: str,
        output_dir: str
    ) -> Dict:
        """
        Extract text from all detected elements.
        
        Args:
            image_path: Path to original image
            detection_json_path: Path to detection results JSON (YOLO already classified)
            output_dir: Directory to save OCR results
            
        Returns:
            OCR results dictionary
        """
        # Load image and detections
        image = cv2.imread(image_path)
        detections = load_json(detection_json_path)
        
        # Extract text from each element
        # Filter out arrowheads (class_id 0) - they dont contain text
        elements = [elem for elem in detections["elements"] if elem.get("class_id") != 0]
        
        # Extract text from each element
        for element in elements:
            bbox = element['bbox']
            x, y, w, h = bbox
            
            # Crop element
            crop = image[y:y+h, x:x+w]
            
            # Extract text
            ocr_result = self.extract_text(crop)
            element['text'] = ocr_result['text']
            element['text_lines'] = ocr_result['lines']
            element['ocr_confidence'] = ocr_result['confidence']
        
        # Save results
        image_name = Path(image_path).stem
        result = {
            'image_path': image_path,
            'image_name': image_name,
            'elements': elements
        }
        
        ensure_dir(output_dir)
        output_path = os.path.join(output_dir, f"{image_name}_ocr.json")
        save_json(result, output_path)
        
        return result
    
    def extract_decision_labels(self, ocr_results: Dict) -> List[Dict]:
        """
        Extract ja/nee decision labels from OCR results.
        
        Args:
            ocr_results: OCR results dictionary with elements
            
        Returns:
            List of label dictionaries with text, bbox, and center
        """
        labels = []
        label_id = 0
        
        for element in ocr_results.get('elements', []):
            text = element.get('text', '').strip().lower()
            
            # Check if text contains ja or nee
            if 'ja' in text or 'nee' in text:
                bbox = element['bbox']
                x, y, w, h = bbox
                
                # Determine which label
                label_text = 'ja' if 'ja' in text else 'nee'
                
                labels.append({
                    'id': label_id,
                    'text': label_text,
                    'bbox': bbox,
                    'center': (x + w // 2, y + h // 2),
                    'element_id': element.get('id')
                })
                label_id += 1
        
        return labels


    def extract_labels_from_image(self, image_path: str) -> List[Dict]:
        """
        Extract ja/nee decision labels directly from the full image.
        These are small labels on arrow lines, not part of detected boxes.
        """
        import cv2
        image = cv2.imread(image_path)
        
        if self.engine_name == 'easyocr':
            results = self.ocr.readtext(image, detail=1, paragraph=False, min_size=10)
            
            labels = []
            for result in results:
                bbox, text, conf = result
                text_lower = text.lower().strip()
                
                # Check for ja/nee variants
                if text_lower in ['ja', 'nee', 'ja?', 'nee?', 'jа', 'пее']:
                    x_center = int((bbox[0][0] + bbox[2][0]) / 2)
                    y_center = int((bbox[0][1] + bbox[2][1]) / 2)
                    labels.append({
                        'text': 'ja' if 'j' in text_lower else 'nee',
                        'center': (x_center, y_center),
                        'confidence': conf,
                        'bbox': bbox
                    })
            
            return labels
        else:
            return []
