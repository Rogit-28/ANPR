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


if __name__ == "__main__":
    list_variants()
