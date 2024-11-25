"""
Script to verify GPU acceleration is working for both PyTorch/YOLO and FFmpeg.
"""
import sys
import subprocess

def check_pytorch_cuda():
    """Check PyTorch CUDA availability."""
    print("=" * 60)
    print("PYTORCH / CUDA CHECK")
    print("=" * 60)
    try:
        import torch
        print(f"Python version: {sys.version}")
        print(f"Torch version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        if cuda_available:
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"CUDNN enabled: {torch.backends.cudnn.enabled}")
            print(f"CUDNN version: {torch.backends.cudnn.version()}")
            
            # Test actual GPU computation
            print("\nTesting GPU computation...")
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.randn(1000, 1000, device='cuda')
            z = torch.matmul(x, y)
            print(f"GPU matrix multiplication test: SUCCESS (result shape: {z.shape})")
        else:
            print("CUDA not available - check your PyTorch installation!")
            print("Install with: pip install torch --index-url https://download.pytorch.org/whl/cu121")
    except ImportError:
        print("Torch not installed.")
    except Exception as e:
        print(f"Error: {e}")


def check_yolo_gpu():
    """Check if YOLO detector uses GPU."""
    print("\n" + "=" * 60)
    print("YOLO DETECTOR GPU CHECK")
    print("=" * 60)
    try:
        sys.path.insert(0, 'src')
        from anpr.core.yolo_detector import YOLODetector
        
        print("Initializing YOLODetector...")
        detector = YOLODetector()
        
        print(f"Device selected: {detector.device}")
        print(f"FP16 enabled: {detector.fp16}")
        print(f"Model loaded: {detector._model_loaded}")
        
        if 'cuda' in detector.device:
            print("SUCCESS: YOLO is using GPU!")
        else:
            print("WARNING: YOLO is using CPU!")
            
        # Run a quick benchmark
        print("\nRunning quick benchmark...")
        metrics = detector.benchmark_performance()
        print(f"Average inference time: {metrics.get('avg_inference_time', 'N/A')}s")
        print(f"FPS: {metrics.get('fps', 'N/A')}")
        print(f"Memory usage: {metrics.get('memory_usage_mb', 'N/A')} MB")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def check_ffmpeg_nvenc():
    """Check FFmpeg NVENC availability."""
    print("\n" + "=" * 60)
    print("FFMPEG NVENC CHECK")
    print("=" * 60)
    try:
        # Check hwaccels
        result = subprocess.run(
            ['ffmpeg', '-hwaccels'],
            capture_output=True,
            text=True,
            timeout=5
        )
        print("Hardware accelerators available:")
        for line in result.stdout.split('\n'):
            if line.strip() and line.strip() != 'Hardware acceleration methods:':
                print(f"  - {line.strip()}")
        
        # Check NVENC encoders
        result = subprocess.run(
            ['ffmpeg', '-encoders'],
            capture_output=True,
            text=True,
            timeout=5
        )
        nvenc_encoders = [line for line in result.stdout.split('\n') if 'nvenc' in line.lower()]
        print("\nNVENC encoders available:")
        for encoder in nvenc_encoders:
            print(f"  {encoder.strip()}")
        
        if nvenc_encoders:
            print("\nSUCCESS: NVENC is available for FFmpeg!")
        else:
            print("\nWARNING: NVENC not available!")
            
    except Exception as e:
        print(f"Error: {e}")


def check_nvidia_smi():
    """Check nvidia-smi output."""
    print("\n" + "=" * 60)
    print("NVIDIA-SMI CHECK")
    print("=" * 60)
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,utilization.gpu', 
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            if len(parts) >= 4:
                print(f"GPU Name: {parts[0]}")
                print(f"Total Memory: {parts[1]} MB")
                print(f"Free Memory: {parts[2]} MB")
                print(f"GPU Utilization: {parts[3]}%")
        else:
            print("nvidia-smi failed")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    check_nvidia_smi()
    check_pytorch_cuda()
    check_yolo_gpu()
    check_ffmpeg_nvenc()
    print("\n" + "=" * 60)
    print("CHECK COMPLETE")
    print("=" * 60)
