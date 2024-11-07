"""
Memory Manager for ANPR System

Provides memory monitoring and optimization for CPU and GPU resources.
"""
import gc
import logging
from typing import Optional
from dataclasses import dataclass

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class MemoryMetrics:
    """Data class to hold memory metrics."""
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    memory_used_gb: float
    gpu_memory_allocated_gb: Optional[float] = None
    gpu_memory_reserved_gb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None


class MemoryManager:
    """
    Manages memory usage for the ANPR system, including GPU memory management,
    memory monitoring, and memory optimization strategies.
    """
    
    def __init__(self, cpu_memory_threshold: float = 0.8, gpu_memory_threshold: float = 0.8):
        """
        Initialize the MemoryManager.
        
        Args:
            cpu_memory_threshold: Threshold for CPU memory usage (0.0-1.0)
            gpu_memory_threshold: Threshold for GPU memory usage (0.0-1.0)
        """
        self.cpu_memory_threshold = cpu_memory_threshold
        self.gpu_memory_threshold = gpu_memory_threshold
        
        # Check if CUDA is available
        self.cuda_available = TORCH_AVAILABLE and torch.cuda.is_available()
        if self.cuda_available:
            logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("GPU not available, will manage CPU memory only")
    
    def get_memory_metrics(self) -> MemoryMetrics:
        """Get current memory usage metrics."""
        # Get CPU memory metrics
        cpu_percent = 0.0
        memory_percent = 0.0
        memory_available_gb = 0.0
        memory_used_gb = 0.0
        
        if PSUTIL_AVAILABLE:
            cpu_percent = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent
            memory_available_gb = memory_info.available / (1024**3)
            memory_used_gb = memory_info.used / (1024**3)
        
        # Get GPU memory metrics if available
        gpu_memory_allocated_gb = None
        gpu_memory_reserved_gb = None
        gpu_utilization_percent = None
        
        if self.cuda_available:
            try:
                gpu_memory_allocated_gb = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_reserved_gb = torch.cuda.memory_reserved() / (1024**3)
                gpu_total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                gpu_utilization_percent = (gpu_memory_allocated_gb / gpu_total_memory_gb) * 100
            except Exception as e:
                logger.warning(f"Error getting GPU metrics: {e}")
        
        return MemoryMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_available_gb=memory_available_gb,
            memory_used_gb=memory_used_gb,
            gpu_memory_allocated_gb=gpu_memory_allocated_gb,
            gpu_memory_reserved_gb=gpu_memory_reserved_gb,
            gpu_utilization_percent=gpu_utilization_percent
        )
    
    def is_memory_usage_high(self) -> bool:
        """Check if memory usage is above threshold."""
        metrics = self.get_memory_metrics()
        
        if metrics.memory_percent / 100 > self.cpu_memory_threshold:
            return True
        
        if self.cuda_available and metrics.gpu_utilization_percent:
            if metrics.gpu_utilization_percent / 100 > self.gpu_memory_threshold:
                return True
        
        return False
    
    def clear_memory_cache(self):
        """Clear memory caches to free up memory."""
        gc.collect()
        
        if self.cuda_available:
            try:
                torch.cuda.empty_cache()
                logger.debug("GPU cache cleared")
            except Exception as e:
                logger.warning(f"Error clearing GPU cache: {e}")
        
        logger.debug("Memory cache cleared")
    
    def optimize_memory_usage(self):
        """Optimize memory usage by clearing caches and checking memory."""
        self.clear_memory_cache()
        
        metrics = self.get_memory_metrics()
        logger.info(f"Memory optimization complete. "
                   f"CPU: {metrics.memory_percent}%, "
                   f"Available: {metrics.memory_available_gb:.2f}GB")
        
        if self.cuda_available and metrics.gpu_memory_allocated_gb is not None:
            logger.info(f"GPU: {metrics.gpu_utilization_percent:.1f}% "
                       f"({metrics.gpu_memory_allocated_gb:.2f}GB allocated)")
    
    def monitor_memory_usage(self) -> bool:
        """
        Monitor memory usage and trigger optimization if needed.
        
        Returns:
            True if memory usage was high and optimization was triggered
        """
        if self.is_memory_usage_high():
            logger.warning("High memory usage detected, optimizing memory...")
            self.optimize_memory_usage()
            return True
        
        return False
    
    def get_recommendations(self) -> list:
        """Get memory optimization recommendations."""
        recommendations = []
        metrics = self.get_memory_metrics()
        
        if metrics.memory_percent > 80:
            recommendations.append("CPU memory usage is high, consider reducing batch size")
        
        if self.cuda_available and metrics.gpu_utilization_percent and metrics.gpu_utilization_percent > 80:
            recommendations.append("GPU memory usage is high, consider reducing model size")
        
        if metrics.memory_available_gb < 1.0:
            recommendations.append("Low available system memory, consider closing other applications")
        
        return recommendations
