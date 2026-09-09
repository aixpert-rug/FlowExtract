"""
Pipeline package for diagram parsing.

Contains all 3 stages:
- Stage 1: Detection (YOLO)
- Stage 2: OCR (Text extraction)
- Stage 3: Connection Derivation (Directed graph)
"""

from .stage1_detector import ElementDetector
from .stage2_ocr import OCREngine
from .stage3_connections import ArrowDetector

__all__ = ['ElementDetector', 'OCREngine', 'ArrowDetector']
