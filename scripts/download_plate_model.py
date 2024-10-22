#!/usr/bin/env python3
"""
Script to download YOLOv11 license plate detection model from HuggingFace.

Model: morsetechlab/yolov11-license-plate-detection
- Fine-tuned on 10,125 license plate images
- 98.13% mAP@50 accuracy
- Available variants: n (nano), s (small), m (medium), l (large), x (extra-large)
"""

import os
import sys
import urllib.request
from pathlib import Path

# HuggingFace model repository
HF_REPO = "morsetechlab/yolov11-license-plate-detection"
HF_BASE_URL = f"https://huggingface.co/{HF_REPO}/resolve/main"

# Available model variants with their sizes and performance characteristics
MODEL_VARIANTS = {
    "n": {
        "filename": "license-plate-finetune-v1n.pt",
        "size_mb": 5.47,
        "description": "Nano - Fastest, lowest accuracy",
        "recommended_for": "Real-time processing, edge devices"
    },
    "s": {
        "filename": "license-plate-finetune-v1s.pt",
        "size_mb": 19.2,
        "description": "Small - Good balance of speed and accuracy",
        "recommended_for": "General use, RTX 4050 (recommended)"
    },
    "m": {
        "filename": "license-plate-finetune-v1m.pt",
        "size_mb": 40.5,
        "description": "Medium - Higher accuracy, moderate speed",
        "recommended_for": "When accuracy is more important than speed"
    },
    "l": {
        "filename": "license-plate-finetune-v1l.pt",
        "size_mb": 51.2,
        "description": "Large - High accuracy",
        "recommended_for": "High-end GPUs with more VRAM"
    },
    "x": {
        "filename": "license-plate-finetune-v1x.pt",
        "size_mb": 114,
        "description": "Extra-Large - Highest accuracy (98.13% mAP@50)",
        "recommended_for": "Maximum accuracy, batch processing"
    }
}


def download_file(url: str, dest_path: Path, show_progress: bool = True) -> bool:
    """Download a file from URL to destination path with progress indicator."""
    try:
        print(f"Downloading from: {url}")
        print(f"Saving to: {dest_path}")
        
        def reporthook(count, block_size, total_size):
            if show_progress and total_size > 0:
                downloaded_mb = count * block_size / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                percent = int(count * block_size * 100 / total_size)
                bar_length = 40
                filled = int(bar_length * percent / 100)
                bar = "=" * filled + "-" * (bar_length - filled)
                sys.stdout.write(f"\r[{bar}] {percent}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, dest_path, reporthook)
        print("\nDownload complete!")
        return True
    except Exception as e:
        print(f"\nError downloading: {e}")
        return False


def verify_model(model_path: Path) -> bool:
    """Verify that the downloaded model loads correctly."""
    try:
        from ultralytics import YOLO
        import numpy as np
        
        print(f"\nVerifying model: {model_path}")
        model = YOLO(str(model_path))
        
        # Create a dummy test image
        test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Run inference
        results = model(test_img, verbose=False)
        
        print(f"Model loaded successfully!")
        print(f"Model classes: {model.names}")
        print(f"Number of classes: {len(model.names)}")
        
        return True
        
    except Exception as e:
        print(f"Error verifying model: {e}")
        return False


def download_license_plate_model(variant: str = "s", models_dir: Path = None) -> Path:
    """
    Download YOLOv11 license plate detection model.
    
    Args:
        variant: Model variant (n, s, m, l, x). Default is 's' (small).
        models_dir: Directory to save the model. Default is project's models/ dir.
    
    Returns:
        Path to the downloaded model file, or None if failed.
    """
    if variant not in MODEL_VARIANTS:
        print(f"Invalid variant '{variant}'. Available: {list(MODEL_VARIANTS.keys())}")
        return None
    
    model_info = MODEL_VARIANTS[variant]
    
    # Determine models directory
    if models_dir is None:
        project_root = Path(__file__).parent.parent
        models_dir = project_root / "models"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Local filename - use a cleaner name
    local_filename = f"yolov11{variant}-license-plate.pt"
    dest_path = models_dir / local_filename
    
    # Check if already downloaded
    if dest_path.exists():
        print(f"Model already exists: {dest_path}")
        if verify_model(dest_path):
            return dest_path
        else:
            print("Existing model is corrupted, re-downloading...")
            dest_path.unlink()
    
    print("=" * 60)
    print("YOLOv11 License Plate Detection Model Download")
    print("=" * 60)
    print(f"Repository: {HF_REPO}")
    print(f"Variant: {variant} ({model_info['description']})")
    print(f"Size: ~{model_info['size_mb']} MB")
    print(f"Recommended for: {model_info['recommended_for']}")
    print("=" * 60)
    print()
    
    # Build download URL
    url = f"{HF_BASE_URL}/{model_info['filename']}"
    
    # Download
    if download_file(url, dest_path):
        if verify_model(dest_path):
            print()
            print("=" * 60)
            print("Download and verification successful!")
            print(f"Model saved to: {dest_path}")
            print("=" * 60)
            return dest_path
        else:
            print("Model verification failed!")
            return None
    else:
        return None


def list_variants():
    """Print available model variants."""
    print("=" * 70)
    print("Available YOLOv11 License Plate Detection Model Variants")
    print("=" * 70)
    print(f"{'Variant':<8} {'Size':<12} {'Description':<45}")
    print("-" * 70)
    for variant, info in MODEL_VARIANTS.items():
        print(f"{variant:<8} {info['size_mb']:<10.1f}MB {info['description']:<45}")
    print("-" * 70)
    print("Recommended: 's' (small) for RTX 4050 - good balance of speed & accuracy")
    print("=" * 70)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download YOLOv11 License Plate Detection Model from HuggingFace"
    )
    parser.add_argument(
        "-v", "--variant",
        type=str,
        default="s",
        choices=list(MODEL_VARIANTS.keys()),
        help="Model variant: n(ano), s(mall), m(edium), l(arge), x(tra-large). Default: s"
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List available model variants"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory for model file"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_variants()
        return 0
    
    output_dir = Path(args.output) if args.output else None
    result = download_license_plate_model(variant=args.variant, models_dir=output_dir)
    
    if result:
        print()
        print("To use this model in the ANPR system, update config.yaml:")
        print(f'  model_path: "{result}"')
        return 0
    else:
        print("\nDownload failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
