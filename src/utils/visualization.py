"""
Visualization utilities for the decomposed parsing pipeline.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple


def draw_bounding_boxes(
    image: np.ndarray,
    boxes: List[Dict],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding boxes on image.
    
    Args:
        image: Input image
        boxes: List of box dictionaries with 'bbox' key containing [x, y, w, h]
        color: BGR color tuple
        thickness: Line thickness
        
    Returns:
        Image with bounding boxes drawn
    """
    vis_image = image.copy()
    
    for box in boxes:
        bbox = box['bbox']
        x, y, w, h = bbox
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, thickness)
        
        # Draw element ID if available
        if 'id' in box:
            cv2.putText(
                vis_image,
                str(box['id']),
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
    
    return vis_image


def draw_element_types(
    image: np.ndarray,
    elements: List[Dict]
) -> np.ndarray:
    """
    Draw bounding boxes with different colors for different element types,
    and fill the background with a solid opaque color to permanently destroy
    legibility of sensitive text information.
    
    Args:
        image: Input image
        elements: List of element dictionaries with 'bbox' and 'type' keys
        
    Returns:
        Image with colored bounding boxes and completely redacted text
    """
    vis_image = image.copy()
    
    # Color mapping for all actual element types in the dataset (pastel fills)
    fill_colors = {
        'process': (200, 255, 200),     # Pastel Green
        'decision': (255, 200, 200),    # Pastel Blue
        'document': (200, 200, 255),    # Pastel Red
        'terminator': (255, 255, 200),  # Pastel Cyan
        'connector': (255, 200, 255)    # Pastel Magenta
    }
    
    # Color mapping for thick borders
    border_colors = {
        'process': (0, 180, 0),         # Solid Green
        'decision': (200, 0, 0),        # Solid Blue
        'document': (0, 0, 200),        # Solid Red
        'terminator': (180, 200, 0),    # Solid Cyan
        'connector': (200, 0, 200)      # Solid Magenta
    }
    
    for element in elements:
        element_type = element.get('type', 'unknown')
        if element_type == 'arrowhead':
            continue
            
        bbox = element['bbox']
        
        fill_color = fill_colors.get(element_type, (220, 220, 220))
        border_color = border_colors.get(element_type, (128, 128, 128))
        
        x, y, w, h = bbox
        
        # 1. Solid opaque fill to permanently destroy/anonymize the underlying text
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), fill_color, cv2.FILLED)
        
        # 2. Draw solid borders on top
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), border_color, 4)
        
        # 3. Draw type label
        label = f"{element.get('id', '?')}: {element_type}"
        font_scale = 1.2
        font_thickness = 3
        
        # Calculate text size for background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        
        # Draw white background box for text to make it extremely readable
        text_y_base = y - 15
        cv2.rectangle(vis_image, (x, text_y_base - th - 5), 
                      (x + tw + 5, text_y_base + 5), (255, 255, 255), -1)
        
        cv2.putText(
            vis_image,
            label,
            (x, text_y_base),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            border_color,
            font_thickness
        )
    
    return vis_image


def draw_arrows(
    image: np.ndarray,
    arrows: List[Dict],
    elements: List[Dict]
) -> np.ndarray:
    """
    Draw arrows between elements from edge to edge with high visibility.
    
    Args:
        image: Input image
        arrows: List of arrow dictionaries with 'source' and 'target' keys
        elements: List of element dictionaries with 'id' and 'bbox' keys
        
    Returns:
        Image with arrows drawn
    """
    vis_image = image.copy()
    
    # Create element ID to bbox mapping
    bboxes = {}
    for element in elements:
        bboxes[element['id']] = element['bbox']
    
    def get_edge_anchors(b1, b2):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        
        c1x, c1y = x1 + w1 // 2, y1 + h1 // 2
        c2x, c2y = x2 + w2 // 2, y2 + h2 // 2
        
        dx = c2x - c1x
        dy = c2y - c1y
        
        if abs(dx) > abs(dy):
            # predominantly horizontal
            if dx > 0:
                return (x1 + w1, c1y), (x2, c2y)
            else:
                return (x1, c1y), (x2 + w2, c2y)
        else:
            # predominantly vertical
            if dy > 0:
                return (c1x, y1 + h1), (c2x, y2)
            else:
                return (c1x, y1), (c2x, y2 + h2)

    # Draw arrows
    for arrow in arrows:
        source_id = arrow['source']
        target_id = arrow['target']
        
        if source_id in bboxes and target_id in bboxes:
            b1 = bboxes[source_id]
            b2 = bboxes[target_id]
            
            start, end = get_edge_anchors(b1, b2)
            
            # Draw highly visible arrow line
            cv2.arrowedLine(
                vis_image,
                start,
                end,
                (255, 0, 255),  # Magenta
                6,              # Extra thick line
                tipLength=0.1   # Slightly longer tip
            )
            
            # Draw label if available
            if 'label' in arrow and arrow['label']:
                mid_x = (start[0] + end[0]) // 2
                mid_y = (start[1] + end[1]) // 2
                
                # Highly visible text background
                text = arrow['label'].upper()
                font_scale = 1.2
                font_thickness = 3
                
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                cv2.rectangle(vis_image, (mid_x - 10, mid_y - th - 10), 
                              (mid_x + tw + 10, mid_y + 10), (255, 255, 255), -1)
                
                # Draw text
                cv2.putText(
                    vis_image,
                    text,
                    (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 0, 255),
                    font_thickness
                )
    
    return vis_image

