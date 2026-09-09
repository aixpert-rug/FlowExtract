"""
Main Pipeline Script

Runs the complete decomposed parsing pipeline end-to-end or stage-by-stage.
"""

import argparse
import os
from pathlib import Path
from typing import Dict

from .utils.io_utils import load_config, get_image_paths
from .pipeline.stage1_detector import ElementDetector
from .pipeline.stage2_ocr import OCREngine
from .pipeline.stage3_connections import ArrowDetector

import shutil

def clear_intermediate_files(intermediate_dir: str, output_dir: str = None, stages: list = None):
    """Clear intermediate and output files before running pipeline.
    
    Args:
        intermediate_dir: Path to intermediate directory
        output_dir: Path to output directory (optional)
        stages: List of stages to clear (1, 2, 3) or None for all
    """
    stage_dirs = {
        1: 'detection',
        2: 'ocr', 
        3: 'arrows'
    }
    
    if stages is None:
        stages = [1, 2, 3]
    
    for stage in stages:
        if stage in stage_dirs:
            dir_path = Path(intermediate_dir) / stage_dirs[stage]
            if dir_path.exists():
                # Remove all files in the directory
                for f in dir_path.glob('*'):
                    if f.is_file():
                        f.unlink()
                print(f"✓ Cleared {stage_dirs[stage]}/")
    
    # Clear output directory if provided and stage 3 is being run
    if output_dir and (stages is None or 3 in stages):
        output_path = Path(output_dir)
        if output_path.exists():
            cleared_count = 0
            for f in output_path.glob('*_arrows.json'):
                f.unlink()
                cleared_count += 1
            if cleared_count > 0:
                print(f"✓ Cleared {cleared_count} files from output/")


def run_stage_1(config: Dict, input_dir: str, output_dir: str):
    """Run Stage 1: Element Detection."""
    print("STAGE 1: ELEMENT DETECTION")
    
    detector = ElementDetector(config)
    results = detector.process_directory(input_dir, output_dir)
    
    print(f"\nDetection complete. Processed {len(results)} images.")
    return results


def run_stage_2(config: Dict, input_dir: str, detection_dir: str, output_dir: str):
    """Run Stage 2: OCR."""
    print("STAGE 2: OCR TEXT EXTRACTION")
    
    ocr_engine = OCREngine(config)
    
    # Get all detection JSON files (YOLO already classified)
    detection_files = list(Path(detection_dir).glob('*_detection.json'))
    
    results = []
    for detection_file in detection_files:
        # Find corresponding image
        image_name = detection_file.stem.replace('_detection', '')
        image_paths = get_image_paths(input_dir)
        image_path = None
        
        for img_path in image_paths:
            if Path(img_path).stem == image_name:
                image_path = img_path
                break
        
        if image_path is None:
            print(f"Warning: Could not find image for {detection_file}")
            continue
        
        result = ocr_engine.process_elements(image_path, str(detection_file), output_dir)
        results.append(result)
        print(f"✓ Extracted text from {image_name}")
    
    print(f"\nOCR complete. Processed {len(results)} images.")
    return results


def run_stage_3(config: Dict, input_dir: str, detection_dir: str, ocr_dir: str, output_dir: str):
    """Run Stage 3: Connection Derivation."""
    print("STAGE 3: CONNECTION DERIVATION")
    
    arrow_detector = ArrowDetector(config)
    ocr_engine = OCREngine(config)  # For label extraction
    
    # Get all OCR JSON files
    ocr_files = list(Path(ocr_dir).glob('*_ocr.json'))
    
    results = []
    for ocr_file in ocr_files:
        # Find corresponding image
        image_name = ocr_file.stem.replace('_ocr', '')
        image_paths = get_image_paths(input_dir)
        image_path = None
        
        for img_path in image_paths:
            if Path(img_path).stem == image_name:
                image_path = img_path
                break
        
        if image_path is None:
            print(f"Warning: Could not find image for {ocr_file}")
            continue
        
        # Load detection data (has arrowheads) and OCR data (has text)
        from .utils.io_utils import load_json
        detection_file = Path(detection_dir) / f"{image_name}_detection.json"
        detection_data = load_json(str(detection_file))
        ocr_data = load_json(str(ocr_file))
        
        # Merge: add text from OCR to detection elements
        ocr_text_map = {e["id"]: e.get("text", "") for e in ocr_data["elements"]}
        for elem in detection_data["elements"]:
            elem["text"] = ocr_text_map.get(elem["id"], "")
        
        # Extract labels from full image (ja/nee text on arrows)
        # This runs OCR on the full image to find small labels
        if hasattr(ocr_engine, 'extract_labels_from_image'):
            labels = ocr_engine.extract_labels_from_image(image_path)
        else:
            # Fallback to old method
            labels = ocr_engine.extract_decision_labels(ocr_data)
        
        # Detect arrows with label assignment (using detection data with text)
        result = arrow_detector.detect_arrows(image_path, detection_data, labels)
        
        # Save result
        from .utils.io_utils import save_json, ensure_dir
        ensure_dir(output_dir)
        output_path = os.path.join(output_dir, f"{image_name}_arrows.json")
        save_json(result, output_path)
        
        results.append(result)
        
        # Count labeled edges
        labeled_edges = sum(1 for e in result['graph']['edges'] if 'label' in e)
        print(f"✓ Detected {result['num_edges']} connections ({labeled_edges} labeled) in {image_name}")
    
    print(f"\nConnection derivation complete. Processed {len(results)} images.")
    return results
        

def main():
    parser = argparse.ArgumentParser(description='Decomposed Parsing Pipeline')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/pipeline_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--stage',
        type=str,
        choices=['all', '1', '2', '3'],
        default='all',
        help='Which stage to run (default: all)'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default=None,
        help='Input directory (overrides config)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get paths from config
    paths_config = config.get('paths', {})
    input_dir = args.input_dir or paths_config.get('input_dir', 'data/input/documents')
    intermediate_dir = paths_config.get('intermediate_dir', 'data/intermediate')
    output_dir = args.output_dir or paths_config.get('output_dir', 'data/output')

    
    # Clear old intermediate and output files before running
    stages_to_clear = [args.stage] if args.stage else None
    clear_intermediate_files(intermediate_dir, output_dir, stages_to_clear)
    
    # Create intermediate subdirectories for each stage
    detection_dir = os.path.join(intermediate_dir, 'detection')
    ocr_dir = os.path.join(intermediate_dir, 'ocr')
    arrows_dir = os.path.join(intermediate_dir, 'arrows')
    
    # Ensure directories exist
    from .utils.io_utils import ensure_dir
    ensure_dir(detection_dir)
    ensure_dir(ocr_dir)
    ensure_dir(arrows_dir)
    ensure_dir(output_dir)
    
    print("DECOMPOSED PARSING PIPELINE (3 STAGES)")
    print(f"Input: {input_dir}")
    print(f"Intermediate: {intermediate_dir}")
    print(f"Output: {output_dir}")
    print(f"Running stage(s): {args.stage}")
    
    # Run selected stage(s)
    if args.stage in ['all', '1']:
        print("RUNNING STAGE 1: ELEMENT DETECTION")
        run_stage_1(config, input_dir, detection_dir)
    
    if args.stage in ['all', '2']:
        print("RUNNING STAGE 2: OCR TEXT EXTRACTION")
        run_stage_2(config, input_dir, detection_dir, ocr_dir)
    
    if args.stage in ['all', '3']:
        print("RUNNING STAGE 3: CONNECTION DERIVATION")
        run_stage_3(config, input_dir, detection_dir, ocr_dir, arrows_dir)
    
    print("PIPELINE COMPLETE")
    print(f"\nIntermediate results: {intermediate_dir}/")
    print(f"  - Detection: {detection_dir}/")
    print(f"  - OCR: {ocr_dir}/")
    print(f"  - Arrows: {arrows_dir}/")
    
    # If all stages run, copy final predictions to output
    if args.stage == 'all':
        print(f"\nFinal predictions will be saved to: {output_dir}/")
        print("(Copying directed graphs from arrows stage...)")
        
        import shutil
        arrow_files = list(Path(arrows_dir).glob('*_arrows.json'))
        for arrow_file in arrow_files:
            dest = os.path.join(output_dir, arrow_file.name)
            shutil.copy(arrow_file, dest)
        
        print(f"✓ Copied {len(arrow_files)} prediction files to {output_dir}/")


if __name__ == '__main__':
    main()
