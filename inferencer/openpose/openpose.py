import os
import cv2
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import time
import logging
import sys
from pathlib import Path
import json

# Assuming the pytorch-openpose code is available in a submodule
sys.path.append("../../sub/pytorch-openpose")
from src.body import Body

try:
    import nvjpeg  # NVIDIA's GPU-accelerated JPEG decoder
    HAS_NVJPEG = True
except ImportError:
    HAS_NVJPEG = False


class OpenPoseInferencer:
    """
    OpenPose pedestrian detector for dashcam images (single GPU version)
    """
    def __init__(
        self,
        model_path: str = '../sub/pytorch-openpose/model/body_pose_model.pth',
        device_id: int = 0,
        batch_size: int = 32,
        confidence_threshold: float = 0.4,
        min_keypoints: int = 4,
        scale_factor: float = 0.5,  # Downscale factor for faster processing
    ):
        """
        Initialize the single-GPU inferencer
        
        Args:
            model_path: Path to the OpenPose model
            device_id: GPU device ID to use
            batch_size: Batch size for processing
            confidence_threshold: Minimum confidence threshold for detected keypoints
            min_keypoints: Minimum number of keypoints to consider a valid pedestrian
            scale_factor: Factor to scale down images for faster processing
        """
        self.model_path = model_path
        self.device_id = device_id
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.min_keypoints = min_keypoints
        self.scale_factor = scale_factor
        
        self.setup_logging()
        self.setup_device()
        self.load_model()

        torch.backends.cudnn.benchmark = True
        self.use_amp = False  # No mixed precision
        
    def setup_logging(self):
        """Configure logging for the inferencer"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            stream=sys.stdout
        )
        self.logger = logging.getLogger("OpenPoseInferencer")
    
    def setup_device(self):
        """Setup GPU device"""
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available! Using CPU instead.")
            self.device = torch.device("cpu")
            return
            
        if self.device_id >= torch.cuda.device_count():
            self.logger.warning(f"GPU {self.device_id} not available. Using GPU 0.")
            self.device_id = 0
            
        self.device = torch.device(f"cuda:{self.device_id}")
        torch.cuda.set_device(self.device_id)
        self.logger.info(f"Using device: {self.device} ({torch.cuda.get_device_name(self.device_id)})")
        
    def load_model(self):
        """Load OpenPose model with TorchScript optimization"""
        self.logger.info(f"Loading OpenPose model from {self.model_path}")
        self.model = Body(self.model_path)
        self.use_amp = False  # Disable mixed precision as it's incompatible with OpenPose
        
        # Try to optimize the model with TorchScript
        try:
            if torch.cuda.is_available():
                self.logger.info("Optimizing model with TorchScript...")
                success = self.model.optimize()
                if (success):
                    self.logger.info("TorchScript optimization successful")
                else:
                    self.logger.info("TorchScript optimization failed, using original model")
        except Exception as e:
            self.logger.warning(f"Error during TorchScript optimization: {e}")
        
        self.logger.info("Model loaded successfully")
        self.warmup()  # Call the warmup method
    
    def warmup(self):
        """Warmup the model with a dummy batch to optimize performance."""
        self.logger.info("Warming up model with a dummy batch...")
        
        # Create a dummy batch
        batch_size = min(4, self.batch_size)  # Small batch for warmup
        dummy_tensor = torch.zeros((batch_size, 3, 368, 368), dtype=torch.float32)
        
        if torch.cuda.is_available():
            dummy_tensor = dummy_tensor.cuda()
            
            # Try to optimize the model with TorchScript
            if hasattr(self.model, 'optimize'):
                success = self.model.optimize()
                if success:
                    self.logger.info("Model optimization successful")
                else:
                    self.logger.info("Model optimization failed, using original model")
        
        # Run a forward pass to warm up the model
        with torch.no_grad():
            try:
                _ = self.model.process_batch(dummy_tensor)
                self.logger.info("Warmup complete")
            except Exception as e:
                self.logger.warning(f"Warmup failed: {e}")
                
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Optimized image preprocessing with GPU acceleration when possible"""
        try:
            if HAS_NVJPEG and image_path.lower().endswith(('.jpg', '.jpeg')):
                # Use GPU-accelerated JPEG decoding
                with open(image_path, 'rb') as f:
                    jpeg_data = f.read()
                img = nvjpeg.decode(jpeg_data)
            else:
                # Fall back to OpenCV
                img = cv2.imread(image_path, cv2.IMREAD_COLOR)
                
            if img is None:
                return None
                
            # Convert BGR to RGB (if using nvjpeg it's already RGB)
            if not HAS_NVJPEG:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
            # Use faster resize method
            if self.scale_factor != 1.0:
                new_size = (int(img.shape[1] * self.scale_factor), int(img.shape[0] * self.scale_factor))
                img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
            
            return img
        except Exception as e:
            self.logger.warning(f"Error processing image {image_path}: {e}")
            return None
    
    def preprocess_batch(self, image_paths: List[str]) -> Tuple[List[np.ndarray], List[str]]:
        """Preprocess multiple images in parallel"""
        from concurrent.futures import ThreadPoolExecutor
        
        images = []
        valid_paths = []
        
        def load_and_preprocess(img_path):
            try:
                img = self.preprocess_image(img_path)
                if img is not None:
                    return img_path, img
                return None
            except Exception as e:
                self.logger.warning(f"Error preprocessing {img_path}: {e}")
                return None
        
        # Use thread pool for parallel I/O
        with ThreadPoolExecutor(max_workers=min(16, len(image_paths))) as executor:
            results = list(executor.map(load_and_preprocess, image_paths))
        
        # Filter out None results and separate paths and images
        for result in results:
            if result is not None:
                path, img = result
                valid_paths.append(path)
                images.append(img)
        
        return images, valid_paths
    
    def is_pedestrian(self, subset: np.ndarray, candidate: np.ndarray) -> bool:
        if subset.size == 0:
            return False
        
        # Vectorize operations for better performance
        if subset.shape[0] > 0:
            # Calculate key metrics for all detections at once
            head_parts = np.array([0, 14, 15, 16, 17])
            torso_parts = np.array([1, 2, 5, 8, 11])
            
            # Count valid parts using vectorized operations
            head_counts = np.sum(subset[:, head_parts] >= 0, axis=1)
            torso_counts = np.sum(subset[:, torso_parts] >= 0, axis=1)
            total_counts = subset[:, -1].astype(int)
            
            # Calculate confidences
            confidences = subset[:, -2] / np.maximum(subset[:, -1], 1)
            
            # Check conditions with vectorized operations
            condition1 = (head_counts >= 1) & (torso_counts >= 2) & (confidences >= self.confidence_threshold)
            condition2 = (torso_counts >= 3) & (total_counts >= self.min_keypoints) & (confidences >= self.confidence_threshold)
            condition3 = (total_counts >= self.min_keypoints + 2) & (confidences >= self.confidence_threshold + 0.1)
            
            # If any detection meets any condition, return True
            if np.any(condition1 | condition2 | condition3):
                return True
                
        return False
    
    def process_batch(self, image_paths: List[str]) -> Dict[str, dict]:
        # Pre-load all images in parallel
        images, valid_paths = self.preprocess_batch(image_paths)
        
        start_time = time.time()
        
        load_time = time.time() - start_time
        self.logger.info(f"Loaded {len(valid_paths)} images in {load_time:.2f}s")
        
        results = {}
        batch_start_time = time.time()
        
        # Process in true batches
        if len(valid_paths) > 0:
            try:
                # Add debug logging
                self.logger.info("Starting batch processing")
                
                # Batch processing using modified Body class
                batch_candidates, batch_subsets = self.process_images_in_batch(images)
                
                self.logger.info(f"Batch processing complete. Got {len(batch_candidates)} candidates and {len(batch_subsets)} subsets")
                
                # Process the results for each image
                for i in range(len(valid_paths)):
                    if i < len(batch_candidates) and i < len(batch_subsets):
                        img_path = valid_paths[i]
                        candidate = batch_candidates[i]
                        subset = batch_subsets[i]
                        
                        # Convert keypoints to original scale if needed
                        if self.scale_factor != 1.0 and isinstance(candidate, np.ndarray) and candidate.size > 0:
                            candidate[:, 0] /= self.scale_factor
                            candidate[:, 1] /= self.scale_factor
                        
                        has_pedestrian = self.is_pedestrian(subset, candidate)
                        num_pedestrians = 0 if subset.size == 0 else len(subset)
                        
                        results[img_path] = {
                            "candidate": candidate,
                            "subset": subset,
                            "is_pedestrian": has_pedestrian,
                            "num_pedestrians": num_pedestrians
                        }
                
            except Exception as e:
                self.logger.error(f"Error in batch processing: {e}")
                # Fall back to one-by-one processing
                for i, (img_path, img) in enumerate(zip(valid_paths, images)):
                    try:
                        candidate, subset = self.model(img)
                        
                        # Convert keypoints to original scale if needed
                        if self.scale_factor != 1.0 and candidate.ndim == 2 and candidate.size > 0:
                            candidate[:, 0] /= self.scale_factor
                            candidate[:, 1] /= self.scale_factor
                        
                        has_pedestrian = self.is_pedestrian(subset, candidate)
                        num_pedestrians = 0 if subset.size == 0 else len(subset)
                        
                        results[img_path] = {
                            "candidate": candidate,
                            "subset": subset,
                            "is_pedestrian": has_pedestrian,
                            "num_pedestrians": num_pedestrians
                        }
                    except Exception as e:
                        self.logger.error(f"Error processing image {img_path}: {e}")
                        results[img_path] = {
                            "candidate": np.array([]),
                            "subset": np.array([]),
                            "is_pedestrian": False,
                            "num_pedestrians": 0
                        }
        
        total_time = time.time() - batch_start_time
        avg_fps = len(valid_paths) / total_time if total_time > 0 else 0
        
        self.logger.info(f"Batch complete: {len(valid_paths)} images processed in {total_time:.2f}s - FPS: {avg_fps:.2f}")
        
        return results

    def process_images_in_batch(self, images):
        """Process multiple images in a single batch with true batching."""
        if not images:
            return [], []
            
        # Create batch tensor
        batch_size = len(images)
        
        # Preprocess all images to have the same dimensions
        processed_images = []
        original_shapes = []
        
        for img in images:
            original_shapes.append((img.shape[0], img.shape[1]))
            # Use a fixed size for all images to ensure efficient batch processing
            target_size = (368, 368)  # Standard OpenPose size
            resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            processed_images.append(resized_img)
        
        # Create batch tensor
        batch_tensor = torch.zeros((batch_size, 3, 368, 368), dtype=torch.float32)
        
        for i, img in enumerate(processed_images):
            # Convert to tensor and normalize
            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            batch_tensor[i] = img_tensor
        
        if torch.cuda.is_available():
            batch_tensor = batch_tensor.cuda()
        
        # Add this before inference
        if torch.cuda.is_available() and hasattr(self, 'use_amp') and self.use_amp:
            batch_tensor = batch_tensor
        
        # Use autocast for mixed precision
        with torch.cuda.amp.autocast(enabled=getattr(self, 'use_amp', False)):
            try:
                candidates, subsets = self.model.process_batch(batch_tensor)
                
                # Make sure we get valid lists even if something goes wrong
                if candidates is None or subsets is None:
                    candidates = [np.array([]) for _ in range(batch_size)]
                    subsets = [np.array([]) for _ in range(batch_size)]
                
                # Ensure we have the right number of results
                if len(candidates) != batch_size or len(subsets) != batch_size:
                    self.logger.warning(f"Expected {batch_size} results, got {len(candidates)} candidates and {len(subsets)} subsets")
                    # Pad results if needed
                    candidates = candidates + [np.array([])] * (batch_size - len(candidates))
                    subsets = subsets + [np.array([])] * (batch_size - len(subsets))
                    
            except Exception as e:
                self.logger.error(f"Error in batch processing: {e}")
                candidates = [np.array([]) for _ in range(batch_size)]
                subsets = [np.array([]) for _ in range(batch_size)]
        
        # FIX: Properly scale the candidates without unpacking error
        scaled_candidates = []
        for i in range(len(candidates)):
            candidate = candidates[i]
            if i < len(original_shapes):  # Ensure we don't go out of bounds
                orig_h, orig_w = original_shapes[i]  # Unpack correctly
                if isinstance(candidate, np.ndarray) and candidate.size > 0:
                    # Scale in single operation with float32 precision
                    if candidate.shape[1] >= 3:  # Ensure we have enough dimensions
                        scale_factors = np.array([orig_w / 368.0, orig_h / 368.0, 1.0])
                        candidate[:, :3] *= scale_factors
                scaled_candidates.append(candidate)
            else:
                scaled_candidates.append(np.array([]))
        
        return scaled_candidates, subsets
    
    def process_image_list(self, image_paths: List[str], output_dir: str = None):
        """Process with periodic saving to avoid data loss"""
        num_images = len(image_paths)
        self.logger.info(f"Starting inference on {num_images} images")
        
        # Optimize memory usage
        self.optimize_memory_usage()
        
        results = {}
        pedestrians_found = 0
        total_pedestrians = 0
        
        # Create checkpoint directory
        if output_dir:
            checkpoint_dir = os.path.join(output_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Process in batches with periodic saving
        save_frequency = min(1000, max(self.batch_size * 10, 100))  # Every ~1000 images
        
        for i in range(0, num_images, self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            batch_results = self.process_batch(batch_paths)
            
            results.update(batch_results)
            
            # Update statistics
            for result in batch_results.values():
                if result['is_pedestrian']:
                    pedestrians_found += 1
                total_pedestrians += result['num_pedestrians']
            
            # Save intermediate results periodically
            if output_dir and (i + self.batch_size) % save_frequency < self.batch_size:
                checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{i+self.batch_size}.json")
                self.logger.info(f"Saving checkpoint at {i+self.batch_size}/{num_images} images")
                
                # Save only the paths and pedestrian status for checkpointing
                checkpoint_data = {
                    "pedestrian_images": [p for p, r in results.items() if r["is_pedestrian"]],
                    "non_pedestrian_images": [p for p, r in results.items() if not r["is_pedestrian"]],
                    "processed_count": len(results),
                    "pedestrians_found": pedestrians_found,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f)
                
            self.logger.info(f"Progress: {min(i + self.batch_size, num_images)}/{num_images} images processed")
        
        # Final save of complete results
        if output_dir:
            self.save_detection_results(results, output_dir)
        
        return results
    
    def save_detection_results(self, results: Dict[str, dict], output_dir: str):
        """
        Save detection results to output directory
        
        Args:
            results: Dictionary mapping image paths to detection results
            output_dir: Directory to save results
        """
        self.logger.info(f"Saving detection results for {len(results)} images to {output_dir}")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Check for existing results
        pedestrian_path = os.path.join(output_dir, "pedestrian_images.txt")
        non_pedestrian_path = os.path.join(output_dir, "non_pedestrian_images.txt")
        
        existing_pedestrians = set()
        existing_non_pedestrians = set()
        
        # Load existing results if they exist
        if os.path.exists(pedestrian_path):
            with open(pedestrian_path, 'r') as f:
                existing_pedestrians = set(line.strip() for line in f if line.strip())
            self.logger.info(f"Found {len(existing_pedestrians)} existing pedestrian entries")
                
        if os.path.exists(non_pedestrian_path):
            with open(non_pedestrian_path, 'r') as f:
                existing_non_pedestrians = set(line.strip() for line in f if line.strip())
            self.logger.info(f"Found {len(existing_non_pedestrians)} existing non-pedestrian entries")
        
        # Group images by pedestrian detection status
        pedestrian_images = list(existing_pedestrians)
        non_pedestrian_images = list(existing_non_pedestrians)
        
        # Add new results
        for img_path, result in results.items():
            basename = os.path.basename(img_path)
            if result['is_pedestrian']:
                if basename not in existing_pedestrians:
                    pedestrian_images.append(basename)
            else:
                if basename not in existing_non_pedestrians:
                    non_pedestrian_images.append(basename)
        
        # Save lists of image names
        with open(pedestrian_path, 'w') as f:
            f.write('\n'.join(pedestrian_images))
        
        with open(non_pedestrian_path, 'w') as f:
            f.write('\n'.join(non_pedestrian_images))
        
        # Save detailed results (keypoints) for images with pedestrians
        import pickle
        import json
        
        # Create subdirectory for detailed results
        details_dir = os.path.join(output_dir, "keypoints")
        os.makedirs(details_dir, exist_ok=True)
        
        # Save detailed results for each image with pedestrians
        saved_count = 0
        for img_path, result in results.items():
            if result['is_pedestrian']:
                # Get filename without extension
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                
                # Convert numpy arrays to lists for JSON serialization
                result_json = {
                    "is_pedestrian": result["is_pedestrian"],
                    "num_pedestrians": result["num_pedestrians"],
                    # Convert numpy arrays to lists
                    "candidate": result["candidate"].tolist() if isinstance(result["candidate"], np.ndarray) else [],
                    "subset": result["subset"].tolist() if isinstance(result["subset"], np.ndarray) else []
                }
                
                # Save as JSON
                json_path = os.path.join(details_dir, f"{base_name}.json")
                with open(json_path, 'w') as f:
                    json.dump(result_json, f)
                
                saved_count += 1
        
        # Also save a summary file with aggregate statistics
        summary_path = os.path.join(output_dir, "summary.json")
        
        # Try loading existing summary if it exists
        total_images = len(results)
        images_with_pedestrians = len([r for r in results.values() if r["is_pedestrian"]])
        images_without_pedestrians = total_images - images_with_pedestrians
        
        if os.path.exists(summary_path):
            try:
                with open(summary_path, 'r') as f:
                    existing_summary = json.load(f)
                    # Add previous counts to current ones
                    total_images += existing_summary.get("total_images", 0)
                    images_with_pedestrians += existing_summary.get("images_with_pedestrians", 0)
                    images_without_pedestrians += existing_summary.get("images_without_pedestrians", 0)
            except Exception as e:
                self.logger.warning(f"Error loading existing summary: {e}")
        
        summary = {
            "total_images": total_images,
            "images_with_pedestrians": images_with_pedestrians,
            "images_without_pedestrians": images_without_pedestrians,
            "detection_parameters": {
                "confidence_threshold": self.confidence_threshold,
                "min_keypoints": self.min_keypoints,
                "scale_factor": self.scale_factor
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Saved results: {len(pedestrian_images)} images with pedestrians, {len(non_pedestrian_images)} without")
        self.logger.info(f"Saved detailed keypoints for {saved_count} images with pedestrians")
    
    def optimize_memory_usage(self):
        """Configure for optimal memory usage"""
        if torch.cuda.is_available():
            # Enable memory caching for faster allocation
            torch.cuda.empty_cache()
            
            # Optimize memory allocation strategy
            torch.cuda.set_per_process_memory_fraction(0.97)  # Use more of available memory
            
            # Enable TF32 on Ampere GPUs (A6000 should support this)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Set benchmark mode for faster convolutions
            torch.backends.cudnn.benchmark = True
            
            # Disable gradient calculation (we're only doing inference)
            torch.set_grad_enabled(False)

