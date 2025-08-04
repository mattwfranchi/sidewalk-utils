import cupy as cp
import numpy as np
from typing import Dict, List, Optional
import sys
sys.path.append('/share/ju/sidewalk_utils')
from utils.logger import get_logger

class GPUHelper:
    """Helper class for GPU operations and memory management."""
    
    def __init__(self):
        self.logger = get_logger("GPUHelper")
        self.setup_gpu_memory_pool()
        
    def setup_gpu_memory_pool(self):
        """Set up GPU memory pool for better memory management."""
        try:
            cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
            self.logger.info("GPU memory pool initialized")
        except Exception as e:
            self.logger.warning(f"GPU memory pool setup failed: {e}")
    
    def cleanup_gpu_memory(self):
        """Clean up GPU memory after large operations."""
        try:
            # Clear CuPy memory pool
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            self.logger.info("GPU memory cleaned up")
        except Exception as e:
            self.logger.warning(f"GPU memory cleanup failed: {e}")
    
    def get_gpu_memory_info(self) -> Optional[Dict]:
        """Get current GPU memory usage information."""
        try:
            mempool = cp.get_default_memory_pool()
            
            total = mempool.get_limit()
            used = mempool.used_bytes()
            free = total - used
            
            self.logger.info(f"GPU Memory: {used/1024**3:.2f}GB used, {free/1024**3:.2f}GB free")
            
            return {
                'used_gb': used / 1024**3,
                'free_gb': free / 1024**3,
                'total_gb': total / 1024**3
            }
        except Exception as e:
            self.logger.warning(f"Could not get GPU memory info: {e}")
            return None 