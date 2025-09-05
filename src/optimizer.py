"""
Performance optimization utilities for the Real-Time Object Detection & Tracking system.
"""

import torch
import cv2
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

from .utils import setup_logger


class ModelOptimizer:
    """
    Model optimization utilities for improved performance.
    """
    
    def __init__(self, model, device: str = "cpu"):
        """
        Initialize model optimizer.
        
        Args:
            model: YOLO model instance
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.logger = setup_logger(__name__)
        
        # Optimization settings
        self.optimization_applied = {
            'half_precision': False,
            'tensorrt': False,
            'onnx': False,
            'quantization': False,
            'compilation': False
        }
    
    def enable_half_precision(self) -> bool:
        """
        Enable half precision (FP16) for faster inference.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.device == "cpu":
                self.logger.warning("Half precision not supported on CPU")
                return False
            
            # Enable half precision in model
            if hasattr(self.model, 'half'):
                self.model.half()
                self.optimization_applied['half_precision'] = True
                self.logger.info("Half precision enabled")
                return True
            else:
                self.logger.warning("Model does not support half precision")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to enable half precision: {e}")
            return False
    
    def optimize_for_inference(self) -> Dict[str, bool]:
        """
        Apply multiple optimizations for inference.
        
        Returns:
            Dictionary of applied optimizations
        """
        optimizations = {}
        
        # Enable half precision if GPU available
        if self.device != "cpu":
            optimizations['half_precision'] = self.enable_half_precision()
        
        # Set model to evaluation mode
        if hasattr(self.model, 'eval'):
            self.model.eval()
            optimizations['eval_mode'] = True
        
        # Disable gradient computation
        if hasattr(self.model, 'requires_grad_'):
            self.model.requires_grad_(False)
            optimizations['no_grad'] = True
        
        # Compile model if PyTorch 2.0+
        if hasattr(torch, 'compile') and torch.__version__ >= "2.0":
            try:
                self.model = torch.compile(self.model)
                optimizations['compilation'] = True
                self.logger.info("Model compiled with torch.compile")
            except Exception as e:
                self.logger.warning(f"Failed to compile model: {e}")
                optimizations['compilation'] = False
        
        self.optimization_applied.update(optimizations)
        return optimizations
    
    def benchmark_optimizations(self, test_frames: int = 100) -> Dict:
        """
        Benchmark different optimization settings.
        
        Args:
            test_frames: Number of test frames
            
        Returns:
            Benchmark results
        """
        self.logger.info("Running optimization benchmark...")
        
        # Create dummy input
        dummy_input = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        results = {}
        
        # Test baseline (no optimizations)
        self.logger.info("Testing baseline performance...")
        baseline_time = self._benchmark_inference(dummy_input, test_frames)
        results['baseline'] = {
            'avg_time_ms': baseline_time,
            'fps': 1000 / baseline_time if baseline_time > 0 else 0
        }
        
        # Test with half precision
        if self.device != "cpu":
            self.logger.info("Testing half precision...")
            if self.enable_half_precision():
                half_time = self._benchmark_inference(dummy_input, test_frames)
                results['half_precision'] = {
                    'avg_time_ms': half_time,
                    'fps': 1000 / half_time if half_time > 0 else 0,
                    'speedup': baseline_time / half_time if half_time > 0 else 1
                }
        
        return results
    
    def _benchmark_inference(self, input_frame: np.ndarray, iterations: int) -> float:
        """
        Benchmark inference time.
        
        Args:
            input_frame: Input frame
            iterations: Number of iterations
            
        Returns:
            Average inference time in milliseconds
        """
        times = []
        
        # Warmup
        for _ in range(10):
            try:
                self.model(input_frame, verbose=False)
            except:
                pass
        
        # Benchmark
        for _ in range(iterations):
            start_time = time.time()
            try:
                self.model(input_frame, verbose=False)
            except:
                pass
            end_time = time.time()
            times.append((end_time - start_time) * 1000)
        
        return np.mean(times) if times else 0.0


class FrameProcessor:
    """
    Frame processing optimizations.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        """
        Initialize frame processor.
        
        Args:
            target_size: Target frame size for processing
        """
        self.target_size = target_size
        self.logger = setup_logger(__name__)
        
        # Pre-allocate arrays for better performance
        self._preallocated_arrays = {}
    
    def resize_frame_optimized(self, frame: np.ndarray) -> np.ndarray:
        """
        Optimized frame resizing with pre-allocated arrays.
        
        Args:
            frame: Input frame
            
        Returns:
            Resized frame
        """
        height, width = frame.shape[:2]
        
        # Use pre-allocated array if available
        key = (height, width, self.target_size[0], self.target_size[1])
        if key not in self._preallocated_arrays:
            self._preallocated_arrays[key] = np.empty(
                (self.target_size[1], self.target_size[0], 3), dtype=np.uint8
            )
        
        # Resize using OpenCV's optimized implementation
        resized = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        return resized
    
    def batch_process_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Process multiple frames in batch for better performance.
        
        Args:
            frames: List of input frames
            
        Returns:
            List of processed frames
        """
        if not frames:
            return []
        
        # Resize all frames
        processed_frames = []
        for frame in frames:
            processed_frame = self.resize_frame_optimized(frame)
            processed_frames.append(processed_frame)
        
        return processed_frames
    
    def apply_preprocessing_pipeline(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply optimized preprocessing pipeline.
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame
        """
        # Convert to RGB if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Assume BGR, convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame
        frame = self.resize_frame_optimized(frame)
        
        # Normalize if needed
        frame = frame.astype(np.float32) / 255.0
        
        return frame


class MemoryOptimizer:
    """
    Memory optimization utilities.
    """
    
    def __init__(self):
        """Initialize memory optimizer."""
        self.logger = setup_logger(__name__)
        self.memory_stats = {}
    
    def clear_gpu_cache(self):
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.debug("GPU cache cleared")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage.
        
        Returns:
            Memory usage statistics in MB
        """
        import psutil
        
        # System memory
        process = psutil.Process()
        system_memory = psutil.virtual_memory()
        
        memory_stats = {
            'system_total_mb': system_memory.total / (1024 * 1024),
            'system_available_mb': system_memory.available / (1024 * 1024),
            'system_used_mb': system_memory.used / (1024 * 1024),
            'process_memory_mb': process.memory_info().rss / (1024 * 1024)
        }
        
        # GPU memory if available
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_stats()
            memory_stats.update({
                'gpu_allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024),
                'gpu_cached_mb': torch.cuda.memory_reserved() / (1024 * 1024),
                'gpu_max_allocated_mb': gpu_memory.get('allocated_bytes.all.peak', 0) / (1024 * 1024)
            })
        
        self.memory_stats = memory_stats
        return memory_stats
    
    def optimize_memory_usage(self):
        """Apply memory optimization techniques."""
        # Clear GPU cache
        self.clear_gpu_cache()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        self.logger.info("Memory optimization applied")


class PerformanceProfiler:
    """
    Performance profiling utilities.
    """
    
    def __init__(self):
        """Initialize performance profiler."""
        self.logger = setup_logger(__name__)
        self.profiles = {}
        self.active_profiles = {}
    
    def start_profile(self, name: str):
        """
        Start profiling a section.
        
        Args:
            name: Profile name
        """
        self.active_profiles[name] = time.time()
    
    def end_profile(self, name: str) -> float:
        """
        End profiling a section.
        
        Args:
            name: Profile name
            
        Returns:
            Elapsed time in seconds
        """
        if name not in self.active_profiles:
            self.logger.warning(f"Profile '{name}' was not started")
            return 0.0
        
        elapsed_time = time.time() - self.active_profiles[name]
        
        if name not in self.profiles:
            self.profiles[name] = []
        
        self.profiles[name].append(elapsed_time)
        del self.active_profiles[name]
        
        return elapsed_time
    
    def get_profile_stats(self, name: str) -> Dict:
        """
        Get profile statistics.
        
        Args:
            name: Profile name
            
        Returns:
            Profile statistics
        """
        if name not in self.profiles or not self.profiles[name]:
            return {}
        
        times = self.profiles[name]
        return {
            'count': len(times),
            'total_time': sum(times),
            'avg_time': np.mean(times),
            'min_time': min(times),
            'max_time': max(times),
            'std_time': np.std(times)
        }
    
    def get_all_stats(self) -> Dict:
        """
        Get statistics for all profiles.
        
        Returns:
            All profile statistics
        """
        return {name: self.get_profile_stats(name) for name in self.profiles}
    
    def reset_profiles(self):
        """Reset all profiles."""
        self.profiles.clear()
        self.active_profiles.clear()
        self.logger.info("All profiles reset")


class OptimizationManager:
    """
    Central manager for all optimization features.
    """
    
    def __init__(self, model, device: str = "cpu"):
        """
        Initialize optimization manager.
        
        Args:
            model: YOLO model instance
            device: Device to run on
        """
        self.model = model
        self.device = device
        
        # Initialize optimizers
        self.model_optimizer = ModelOptimizer(model, device)
        self.frame_processor = FrameProcessor()
        self.memory_optimizer = MemoryOptimizer()
        self.profiler = PerformanceProfiler()
        
        self.logger = setup_logger(__name__)
    
    def apply_all_optimizations(self) -> Dict[str, bool]:
        """
        Apply all available optimizations.
        
        Returns:
            Dictionary of applied optimizations
        """
        self.logger.info("Applying all optimizations...")
        
        optimizations = {}
        
        # Model optimizations
        model_opts = self.model_optimizer.optimize_for_inference()
        optimizations.update(model_opts)
        
        # Memory optimizations
        self.memory_optimizer.optimize_memory_usage()
        optimizations['memory_optimized'] = True
        
        self.logger.info(f"Applied optimizations: {optimizations}")
        return optimizations
    
    def benchmark_system(self, test_frames: int = 100) -> Dict:
        """
        Run comprehensive system benchmark.
        
        Args:
            test_frames: Number of test frames
            
        Returns:
            Comprehensive benchmark results
        """
        self.logger.info("Running comprehensive system benchmark...")
        
        results = {}
        
        # Model benchmark
        results['model'] = self.model_optimizer.benchmark_optimizations(test_frames)
        
        # Memory benchmark
        results['memory'] = self.memory_optimizer.get_memory_usage()
        
        # Frame processing benchmark
        dummy_frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(10)
        ]
        
        start_time = time.time()
        processed_frames = self.frame_processor.batch_process_frames(dummy_frames)
        end_time = time.time()
        
        results['frame_processing'] = {
            'frames_processed': len(processed_frames),
            'total_time': end_time - start_time,
            'avg_time_per_frame': (end_time - start_time) / len(processed_frames)
        }
        
        return results
    
    def get_optimization_report(self) -> Dict:
        """
        Get comprehensive optimization report.
        
        Returns:
            Optimization report
        """
        return {
            'model_optimizations': self.model_optimizer.optimization_applied,
            'memory_usage': self.memory_optimizer.get_memory_usage(),
            'profile_stats': self.profiler.get_all_stats(),
            'device': self.device,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available()
        }
