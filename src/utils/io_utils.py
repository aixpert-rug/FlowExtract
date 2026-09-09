"""
Utility functions for the decomposed parsing pipeline.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
import yaml


def load_config(config_path: str = "configs/pipeline_config.yaml") -> Dict:
    """Load pipeline configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_json(data: Any, output_path: str) -> None:
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(json_path: str) -> Any:
    """Load data from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_image_paths(input_dir: str) -> List[str]:
    """Get all image paths from input directory."""
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(Path(input_dir).rglob(f'*{ext}'))
        image_paths.extend(Path(input_dir).rglob(f'*{ext.upper()}'))
    
    return sorted([str(p) for p in image_paths])


def ensure_dir(directory: str) -> None:
    """Ensure directory exists."""
    os.makedirs(directory, exist_ok=True)
