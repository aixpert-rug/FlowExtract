import os
import json
import cv2
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.visualization import draw_element_types, draw_arrows

def generate_qualitative_figure(json_path, output_path):
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Load Image
    image_path = data.get('image_path')
    if not image_path or not os.path.exists(image_path):
        # Fallback to absolute path relative to project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        image_name = data.get('image_name')
        if not image_name:
            image_name = Path(json_path).stem.replace('_arrows', '')
        # Try to find it in train or test
        img_paths = [
            os.path.join(project_root, 'data', 'input', 'documents', 'test', f"{image_name}.png"),
            os.path.join(project_root, 'data', 'input', 'documents', 'train', f"{image_name}.png")
        ]
        img_file = next((p for p in img_paths if os.path.exists(p)), None)
        if not img_file:
            print(f"Error: Could not find image for {json_path}")
            return
    else:
        img_file = os.path.abspath(image_path)
        
    img = cv2.imread(img_file)
    if img is None:
        print(f"Error: Could not load image {img_file}")
        return
        
    print(f"Loaded image: {img_file}")
    
    nodes = data['graph']['nodes']
    edges = data['graph']['edges']
    
    # Draw boxes
    img_boxes = draw_element_types(img, nodes)
    
    # Draw arrows
    img_arrows = draw_arrows(img_boxes, edges, nodes)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img_arrows)
    print(f"Saved qualitative figure to: {output_path}")

if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Run Example 1
    json_path1 = os.path.join(project_root, 'data', 'output', 'example1_arrows.json')
    output_path1 = os.path.join(project_root, 'data', 'output', 'charts', 'qualitative_example1.png')
    generate_qualitative_figure(json_path1, output_path1)
    
    # Run Example 2
    json_path2 = os.path.join(project_root, 'data', 'output', 'example2_arrows.json')
    output_path2 = os.path.join(project_root, 'data', 'output', 'charts', 'qualitative_example2.png')
    generate_qualitative_figure(json_path2, output_path2)
