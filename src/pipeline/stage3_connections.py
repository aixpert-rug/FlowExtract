"""
Stage 3: Connection Derivation

Derives directed graph connections using line tracing and proximity matching.
Outputs nodes (elements) and edges (leads_to relations).
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import os
from pathlib import Path

from ..utils.io_utils import load_json, save_json, ensure_dir



# YOLO class ID to type name mapping (from yolo_data.yaml)
YOLO_CLASS_NAMES = {
    0: 'arrowhead',
    1: 'connector',
    2: 'decision',
    3: 'document',
    4: 'process',
    5: 'terminator'
}

class ArrowDetector:
    """Detector for arrow connections using line tracing and proximity matching."""
    
    def __init__(self, config: Dict):
        """
        Initialize arrow detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        arrow_config = config.get('arrows', {})
        
        # Line detection parameters
        line_config = arrow_config.get('line_detection', {})
        self.hough_threshold = line_config.get('hough_threshold', 50)
        self.min_line_length = line_config.get('min_line_length', 20)
        self.max_line_gap = line_config.get('max_line_gap', 15)
        
        # Connection matching parameters
        self.line_attachment_threshold = arrow_config.get('line_attachment_threshold', 15)
        self.target_touch_threshold = arrow_config.get('target_touch_threshold', 20)
        self.source_trace_threshold = arrow_config.get('source_trace_threshold', 30)
        self.junction_detection_threshold = arrow_config.get('junction_detection_threshold', 15)
        
        # Label assignment parameters
        label_config = arrow_config.get('label_assignment', {})
        self.single_arrow_radius = label_config.get('single_arrow_radius', 100)
        self.branching_radius = label_config.get('branching_radius', 150)
        self.distance_penalty_factor = label_config.get('distance_penalty_factor', 50.0)

    def _identify_arrowhead_ends(
        self,
        arrowhead: Dict,
        image: np.ndarray,
        boxes: List[Dict] = None
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Identify blunt and pointy ends of arrowhead.
        
        The pointy end is the one closest to a box (the target).
        The blunt end is where the line comes from (the source direction).
        
        Args:
            arrowhead: Arrowhead element dict with bbox
            image: Input image
            boxes: List of detected boxes (for orientation detection)
            
        Returns:
            (blunt_end, pointy_end) as (x, y) tuples
        """
        x, y, w, h = arrowhead['bbox']
        center = (x + w // 2, y + h // 2)
        
        # Get all 4 cardinal points around the arrowhead
        candidates = [
            (x, center[1], 'left'),           # Left edge center
            (x + w, center[1], 'right'),      # Right edge center
            (center[0], y, 'top'),            # Top edge center
            (center[0], y + h, 'bottom')      # Bottom edge center
        ]
        
        # If we have boxes, find which end is closest to a box (that's the pointy end)
        if boxes:
            best_pointy = None
            min_dist_to_box = float('inf')
            
            for px, py, direction in candidates:
                for box in boxes:
                    bx, by, bw, bh = box['bbox']
                    # Distance from point to box edge
                    dx = max(bx - px, 0, px - (bx + bw))
                    dy = max(by - py, 0, py - (by + bh))
                    dist = np.sqrt(dx**2 + dy**2)
                    
                    if dist < min_dist_to_box:
                        min_dist_to_box = dist
                        best_pointy = (px, py, direction)
            
            if best_pointy and min_dist_to_box < 50:  # Only if reasonably close
                px, py, pointy_dir = best_pointy
                # Blunt is opposite of pointy
                opposites = {'left': 'right', 'right': 'left', 'top': 'bottom', 'bottom': 'top'}
                blunt_dir = opposites[pointy_dir]
                blunt = next((c[0], c[1]) for c in candidates if c[2] == blunt_dir)
                return blunt, (px, py)
        
        # Fallback to aspect ratio heuristic
        if w > h:  # Horizontal: pointy on right
            return (x, center[1]), (x + w, center[1])
        else:  # Vertical: pointy on bottom
            return (center[0], y), (center[0], y + h)
    
    def _find_box_touching_point(
        self,
        point: Tuple[int, int],
        boxes: List[Dict],
        threshold: int = 20
    ) -> Optional[int]:
        """
        Find box whose boundary is closest to point.
        
        Args:
            point: (x, y) coordinates
            boxes: List of element boxes
            threshold: Maximum distance in pixels
            
        Returns:
            Element ID or None
        """
        px, py = point
        min_dist = float('inf')
        closest_box = None
        
        for box in boxes:
            x, y, w, h = box['bbox']
            
            # Calculate distance to box boundary
            dx = max(x - px, 0, px - (x + w))
            dy = max(y - py, 0, py - (y + h))
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist < min_dist and dist <= threshold:
                min_dist = dist
                closest_box = box['id']
        
        return closest_box
    
    def _find_lines_at_point(
        self,
        point: Tuple[int, int],
        line_segments: List[Tuple[int, int, int, int]],
        threshold: int = 15
    ) -> List[Tuple[int, int, int, int]]:
        """
        Find line segments with endpoint near point.
        
        Args:
            point: (x, y) coordinates
            line_segments: List of line segments
            threshold: Distance threshold
            
        Returns:
            List of line segments
        """
        px, py = point
        attached_lines = []
        
        for line in line_segments:
            x1, y1, x2, y2 = line
            dist1 = np.sqrt((x1 - px)**2 + (y1 - py)**2)
            dist2 = np.sqrt((x2 - px)**2 + (y2 - py)**2)
            
            if dist1 <= threshold or dist2 <= threshold:
                attached_lines.append(line)
        
        return attached_lines
    
    def _get_far_endpoint(
        self,
        line: Tuple[int, int, int, int],
        near_point: Tuple[int, int]
    ) -> Tuple[int, int]:
        """
        Get the endpoint of line that's far from near_point.
        
        Args:
            line: (x1, y1, x2, y2)
            near_point: (x, y)
            
        Returns:
            (x, y) of far endpoint
        """
        x1, y1, x2, y2 = line
        nx, ny = near_point
        
        dist1 = np.sqrt((x1 - nx)**2 + (y1 - ny)**2)
        dist2 = np.sqrt((x2 - nx)**2 + (y2 - ny)**2)
        
        return (x2, y2) if dist1 < dist2 else (x1, y1)
    
    def _trace_line_to_sources(
        self,
        line: Tuple[int, int, int, int],
        start_point: Tuple[int, int],
        line_segments: List[Tuple[int, int, int, int]],
        boxes: List[Dict]
    ) -> List[int]:
        """
        Trace line back to source box(es), handling junctions.
        
        Handles both:
        - Simple case: line directly connects to box
        - Junction case: multiple lines merge before arrowhead
        
        Args:
            line: Starting line segment
            start_point: Point where line attaches to arrowhead
            line_segments: All detected line segments
            boxes: List of element boxes
            
        Returns:
            List of source box IDs
        """
        sources = []
        visited_points = set()
        points_to_explore = [self._get_far_endpoint(line, start_point)]
        explored_lines = {id(line)}  # Track explored lines to avoid loops
        
        while points_to_explore:
            current_point = points_to_explore.pop(0)
            
            # Avoid revisiting points
            point_key = (round(current_point[0]), round(current_point[1]))
            if point_key in visited_points:
                continue
            visited_points.add(point_key)
            
            # Check if this point touches a box
            box_id = self._find_box_touching_point(
                current_point, boxes, threshold=self.source_trace_threshold
            )
            
            if box_id is not None:
                # Found a source box!
                if box_id not in sources:
                    sources.append(box_id)
            else:
                # Not at a box - might be a junction point
                # Find all lines connected to this point
                connected_lines = self._find_lines_at_point(
                    current_point, line_segments, threshold=self.junction_detection_threshold
                )
                
                # Explore each connected line we haven't seen yet
                for connected_line in connected_lines:
                    line_id = id(connected_line)
                    if line_id not in explored_lines:
                        explored_lines.add(line_id)
                        far_end = self._get_far_endpoint(connected_line, current_point)
                        points_to_explore.append(far_end)
        
        return sources
    
    def _mask_elements(self, image: np.ndarray, elements: List[Dict]) -> np.ndarray:
        """Create mask with elements removed."""
        mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
        
        for element in elements:
            x, y, w, h = element['bbox']
            # Add padding to ensure complete masking
            padding = 5
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            mask[y1:y2, x1:x2] = 0
        
        return mask
    
    def _detect_lines(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """Detect line segments in masked image."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply mask
        masked = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Edge detection
        edges = cv2.Canny(masked, 50, 150, apertureSize=3)
        
        # Hough line detection
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        if lines is None:
            return []
        
        # Convert to list of tuples
        line_segments = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_segments.append((int(x1), int(y1), int(x2), int(y2)))
        
        return line_segments
    
    def _get_center(self, bbox: List[int]) -> np.ndarray:
        """Get center point of bounding box."""
        x, y, w, h = bbox
        return np.array([x + w / 2, y + h / 2])
    
    def _calculate_path_midpoint(
        self,
        line_segments: List[Tuple[int, int, int, int]],
        blunt_end: Tuple[int, int]
    ) -> Tuple[int, int]:
        """
        Calculate midpoint of traced line path.
        
        Args:
            line_segments: List of line segments forming the path
            blunt_end: Starting point of the path
            
        Returns:
            (x, y) midpoint coordinates
        """
        if not line_segments:
            return blunt_end
        
        # For simplicity, use midpoint of first line segment
        # TODO: Could trace full path and find actual midpoint
        line = line_segments[0]
        x1, y1, x2, y2 = line
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def _find_labels_near_node(
        self,
        node: Dict,
        labels: List[Dict],
        radius: int = 150
    ) -> List[Dict]:
        """
        Find labels within radius of node center.
        
        Args:
            node: Node dictionary with bbox
            labels: List of label dictionaries
            radius: Search radius in pixels
            
        Returns:
            List of labels within radius
        """
        node_center = self._get_center(node['bbox'])
        nearby_labels = []
        
        for label in labels:
            label_center = np.array(label['center'])
            dist = np.linalg.norm(label_center - node_center)
            
            if dist <= radius:
                nearby_labels.append(label)
        
        return nearby_labels
    
    def _calculate_arrow_direction(
        self,
        source_bbox: List[int],
        target_bbox: List[int]
    ) -> np.ndarray:
        """
        Calculate normalized direction vector from source to target.
        
        Args:
            source_bbox: Source node bounding box
            target_bbox: Target node bounding box
            
        Returns:
            Normalized direction vector
        """
        source_center = self._get_center(source_bbox)
        target_center = self._get_center(target_bbox)
        
        direction = target_center - source_center
        norm = np.linalg.norm(direction)
        
        if norm == 0:
            return np.array([0, 0])
        
        return direction / norm
    
    def _distance_to_line(
        self,
        point: Tuple[int, int],
        line_start: np.ndarray,
        line_end: np.ndarray
    ) -> float:
        """
        Calculate perpendicular distance from point to line segment.
        
        Args:
            point: Point coordinates
            line_start: Line start point
            line_end: Line end point
            
        Returns:
            Distance in pixels
        """
        point = np.array(point)
        
        # Vector from start to end
        line_vec = line_end - line_start
        line_len = np.linalg.norm(line_vec)
        
        if line_len == 0:
            return np.linalg.norm(point - line_start)
        
        # Normalized line vector
        line_unit = line_vec / line_len
        
        # Vector from start to point
        point_vec = point - line_start
        
        # Project point onto line
        projection = np.dot(point_vec, line_unit)
        
        # Clamp to line segment
        projection = max(0, min(line_len, projection))
        
        # Closest point on line
        closest = line_start + projection * line_unit
        
        # Distance to closest point
        return np.linalg.norm(point - closest)
    
    def _calculate_alignment_score(
        self,
        arrow: Dict,
        label: Dict,
        source_node: Dict,
        target_node: Dict
    ) -> float:
        """
        Calculate alignment score for (arrow, label) pair.
        
        Args:
            arrow: Arrow dictionary
            label: Label dictionary
            source_node: Source node dictionary
            target_node: Target node dictionary
            
        Returns:
            Alignment score (higher is better)
        """
        # Get arrow direction
        arrow_dir = self._calculate_arrow_direction(
            source_node['bbox'],
            target_node['bbox']
        )
        
        # Get label direction from source
        source_center = self._get_center(source_node['bbox'])
        label_center = np.array(label['center'])
        label_vec = label_center - source_center
        label_norm = np.linalg.norm(label_vec)
        
        if label_norm == 0:
            return -999  # Label at source center, invalid
        
        label_dir = label_vec / label_norm
        
        # Direction alignment (dot product)
        direction_score = np.dot(arrow_dir, label_dir)
        
        # Distance to arrow path
        target_center = self._get_center(target_node['bbox'])
        dist = self._distance_to_line(
            label['center'],
            source_center,
            target_center
        )
        
        distance_penalty = dist / self.distance_penalty_factor
        
        return direction_score - distance_penalty
    

    def _assign_labels_simple(
        self,
        arrows: List[Dict],
        elements: List[Dict],
        labels: List[Dict]
    ) -> Dict[int, str]:
        """
        Assign labels to arrows using direction-based matching.
        
        In troubleshooting diagrams:
        - "nee" arrows typically go RIGHT (to action boxes)
        - "ja" arrows typically go DOWN (to next decision)
        
        We use both distance to midpoint AND direction alignment.
        """
        assignments = {}
        used_labels = set()
        
        for idx, arrow in enumerate(arrows):
            source_node = next((e for e in elements if e['id'] == arrow['source']), None)
            target_node = next((e for e in elements if e['id'] == arrow['target']), None)
            
            if not source_node or not target_node:
                continue
            
            # Calculate arrow direction
            src_center = self._get_center(source_node['bbox'])
            tgt_center = self._get_center(target_node['bbox'])
            arrow_vec = tgt_center - src_center
            arrow_length = np.linalg.norm(arrow_vec)
            
            if arrow_length == 0:
                continue
            
            arrow_dir = arrow_vec / arrow_length
            midpoint = (src_center + tgt_center) / 2
            
            # Score each label
            best_label = None
            best_score = -float('inf')
            best_idx = -1
            
            for i, label in enumerate(labels):
                if i in used_labels:
                    continue
                
                label_center = np.array(label['center'])
                
                # Distance to midpoint (closer is better)
                dist_to_mid = np.linalg.norm(label_center - midpoint)
                
                # Skip if too far from arrow
                if dist_to_mid > 150:
                    continue
                
                # Direction alignment: label should be in direction of arrow
                label_vec = label_center - src_center
                label_dist = np.linalg.norm(label_vec)
                
                if label_dist > 0:
                    label_dir = label_vec / label_dist
                    alignment = np.dot(arrow_dir, label_dir)  # 1.0 = same direction, -1.0 = opposite
                else:
                    alignment = 0
                
                # Score = -distance + alignment_bonus
                # Normalize distance to 0-1 range (assuming max relevant distance is 150px)
                dist_score = 1 - (dist_to_mid / 150)
                score = dist_score * 0.6 + alignment * 0.4  # 60% distance, 40% direction
                
                if score > best_score:
                    best_score = score
                    best_label = label
                    best_idx = i
            
            if best_label and best_score > 0.3:  # Threshold to avoid bad matches
                assignments[idx] = best_label['text']
                used_labels.add(best_idx)
        
        return assignments

    def _assign_labels_to_arrows(
        self,
        arrows: List[Dict],
        elements: List[Dict],
        labels: List[Dict]
    ) -> Dict[int, str]:
        """
        Assign labels to arrows using direction-based matching.
        
        Args:
            arrows: List of arrow dictionaries
            elements: List of element dictionaries
            labels: List of label dictionaries (ja/nee)
            
        Returns:
            Dictionary mapping arrow index to label text
        """
        if not labels:
            return {}
        
        # Group arrows by source
        by_source = {}
        for idx, arrow in enumerate(arrows):
            source_id = arrow['source']
            if source_id not in by_source:
                by_source[source_id] = []
            by_source[source_id].append((idx, arrow))
        
        assignments = {}
        
        for source_id, source_arrows in by_source.items():
            # Get source node
            source_node = next((e for e in elements if e['id'] == source_id), None)
            if not source_node:
                continue
            
            if len(source_arrows) == 1:
                # Simple case: single arrow, find nearest label
                idx, arrow = source_arrows[0]
                target_node = next((e for e in elements if e['id'] == arrow['target']), None)
                
                if target_node:
                    # Find labels near source
                    candidates = self._find_labels_near_node(
                        source_node, labels, radius=self.single_arrow_radius
                    )
                    
                    if candidates:
                        # Get closest to arrow path
                        best_label = None
                        best_score = -999
                        
                        for label in candidates:
                            score = self._calculate_alignment_score(
                                arrow, label, source_node, target_node
                            )
                            if score > best_score:
                                best_score = score
                                best_label = label
                        
                        if best_label:
                            assignments[idx] = best_label['text']
            else:
                # Branching: direction-based assignment
                candidates = self._find_labels_near_node(
                    source_node, labels, radius=self.branching_radius
                )
                
                if not candidates:
                    continue
                
                # Score all (arrow, label) pairs
                scores = []
                for idx, arrow in source_arrows:
                    target_node = next((e for e in elements if e['id'] == arrow['target']), None)
                    if not target_node:
                        continue
                    
                    for label in candidates:
                        score = self._calculate_alignment_score(
                            arrow, label, source_node, target_node
                        )
                        scores.append((score, idx, label))
                
                # Greedy assignment (best score first)
                scores.sort(reverse=True, key=lambda x: x[0])
                used_labels = set()
                
                for score, idx, label in scores:
                    if label['id'] not in used_labels and idx not in assignments:
                        assignments[idx] = label['text']
                        used_labels.add(label['id'])
        
        return assignments
    
    def process_image(
        self,
        image_path: str,
        detection_json_path: str,
        output_dir: str,
        labels: List[Dict] = None,
        save_visualization: bool = True
    ) -> Dict:
        """
        Process single image and detect arrows.
        
        Args:
            image_path: Path to input image
            detection_json_path: Path to detection results JSON
            output_dir: Directory to save outputs
            labels: Optional list of decision labels (ja/nee)
            save_visualization: Whether to save visualization image
            
        Returns:
            Arrow detection results
        """
        # Load detection results
        detection_data = load_json(detection_json_path)
        
        # Detect arrows
        result = self.detect_arrows(image_path, detection_data, labels)
        
        # Save JSON
        ensure_dir(output_dir)
        image_name = Path(image_path).stem
        json_path = os.path.join(output_dir, f"{image_name}_arrows.json")
        save_json(result, json_path)
        
        # Save visualization
        if save_visualization:
            from ..utils.visualization import draw_element_types, draw_arrows
            image = cv2.imread(image_path)
            vis_image = draw_element_types(image, detection_data['elements'])
            vis_image = draw_arrows(vis_image, result['graph']['edges'], detection_data['elements'])
            vis_path = os.path.join(output_dir, f"{image_name}_arrows_vis.png")
            cv2.imwrite(vis_path, vis_image)
        
        return result

    def detect_arrows(
        self,
        image_path: str,
        elements_data: Dict,
        labels: List[Dict] = None
    ) -> Dict:
        """
        Detect arrows using line tracing + proximity matching.
        
        Args:
            image_path: Path to input image
            elements_data: Detection results with elements and arrowheads
            labels: List of decision labels (ja/nee) from OCR
            
        Returns:
            Arrow detection results
        """
        # Load image and separate elements
        image = cv2.imread(image_path)
        elements = elements_data['elements']
        
        arrowheads = [e for e in elements if e.get('class_id') == 0]  # class 0 = arrowhead
        boxes = [e for e in elements if e.get('class_id') != 0]  # All except arrowheads
        
        # Create mask for boxes (exclude arrowheads from mask)
        element_mask = self._mask_elements(image, boxes)
        
        # Detect line segments
        line_segments = self._detect_lines(image, element_mask)
        
        # Process each arrowhead
        arrows = []
        
        for arrowhead in arrowheads:
            # 1. Identify arrowhead ends
            blunt_end, pointy_end = self._identify_arrowhead_ends(arrowhead, image, boxes)
            
            # 2. Find target (box touching pointy end) - PROXIMITY
            target_id = self._find_box_touching_point(
                pointy_end, boxes, threshold=self.target_touch_threshold
            )
            
            if target_id is None:
                continue  # Skip if no target found
            
            # 3. Find source(s) - LINE TRACING with junction detection
            lines_at_blunt = self._find_lines_at_point(
                blunt_end, line_segments, threshold=self.line_attachment_threshold
            )
            
            if not lines_at_blunt:
                # No lines found - might be detection issue
                continue
            
            # Trace each line at blunt end (usually just one, but could be multiple)
            all_sources = []
            for line in lines_at_blunt:
                sources = self._trace_line_to_sources(
                    line, blunt_end, line_segments, boxes
                )
                all_sources.extend(sources)
            
            # Remove duplicates
            all_sources = list(set(all_sources))
            
            # Create arrow for each source → target connection
            for source_id in all_sources:
                # Skip self-loops (node connecting to itself)
                if source_id == target_id:
                    continue
                
                arrows.append({
                    'source': source_id,
                    'target': target_id,
                    'arrowhead_id': arrowhead['id']
                })
        
        # Assign labels to arrows using simple midpoint-based matching.
        # Labels are typically ON the arrow line, so we find the closest label to
        # each arrow's midpoint. Edges are proposed only where an arrowhead was
        # detected; connectivity is never inferred from bare line segments.
        label_assignments = {}

        if labels:
            label_assignments = self._assign_labels_simple(arrows, boxes, labels)
        
        # Prepare output as directed graph
        image_name = Path(image_path).stem
        
        # Create nodes (exclude arrowheads - class 5)
        nodes = []
        for element in boxes:
            nodes.append({
                'id': element['id'],
                'type': YOLO_CLASS_NAMES.get(element.get('class_id'), 'unknown'),
                'text': element.get('text', ''),
                'bbox': element['bbox']
            })
        
        # Create edges with labels
        edges = []
        for idx, arrow in enumerate(arrows):
            edge = {
                'source': arrow['source'],
                'target': arrow['target'],
                'type': 'leads_to'
            }
            # Add label if assigned
            if idx in label_assignments:
                edge['label'] = label_assignments[idx]
            edges.append(edge)
        
        result = {
            'image_path': image_path,
            'image_name': image_name,
            'graph': {
                'nodes': nodes,
                'edges': edges
            },
            'num_nodes': len(nodes),
            'num_edges': len(edges)
        }
        
        return result
