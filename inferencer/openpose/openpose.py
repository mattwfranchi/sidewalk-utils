import os
import cv2
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import time
import logging
import sys
from pathlib import Path

# Assuming the pytorch-openpose code is available in a submodule
sys.path.append("../../sub/pytorch-openpose")
from src.body import Body


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
        """Load OpenPose model with half precision for faster inference"""
        self.logger.info(f"Loading OpenPose model from {self.model_path}")
        self.model = Body(self.model_path)
        
        # Try to convert model to half precision (FP16) for faster computation
        if torch.cuda.is_available() and torch.cuda.get_device_capability(self.device_id)[0] >= 7:
            try:
                # This will need modification in the Body class implementation
                # to fully support FP16 inference
                self.logger.info("Attempting to use FP16 for faster inference")
                # self.model = self.model.half()
            except Exception as e:
                self.logger.warning(f"Could not convert to FP16: {e}")
                
        self.logger.info("Model loaded successfully")
                
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess an image for inference"""
        img = cv2.imread(image_path)
        if img is None:
            self.logger.error(f"Failed to load image: {image_path}")
            return None
            
        # Resize for faster processing
        if self.scale_factor != 1.0:
            height, width = img.shape[:2]
            new_height = int(height * self.scale_factor)
            new_width = int(width * self.scale_factor)
            img = cv2.resize(img, (new_width, new_height))
            
        return img
    
    def is_pedestrian(self, subset: np.ndarray, candidate: np.ndarray) -> bool:
        """More robust pedestrian detection that handles partial occlusion better"""
        if subset.size == 0:
            return False
        
        # OpenPose keypoint indices
        HEAD_PARTS = [0, 14, 15, 16, 17]  # Nose, eyes, ears
        TORSO_PARTS = [1, 2, 5, 8, 11]    # Neck, shoulders, hips
        
        # Check each person detection
        for person_idx in range(len(subset)):
            person_keypoints = subset[person_idx, :18]
            
            head_detected = sum(1 for idx in HEAD_PARTS if person_keypoints[idx] >= 0)
            torso_detected = sum(1 for idx in TORSO_PARTS if person_keypoints[idx] >= 0)
            total_detected = int(subset[person_idx, -1])
            
            confidence = subset[person_idx, -2] / subset[person_idx, -1] if subset[person_idx, -1] > 0 else 0
            
            if (head_detected >= 1 and torso_detected >= 2 and confidence >= self.confidence_threshold):
                return True
            elif (torso_detected >= 3 and total_detected >= self.min_keypoints and 
                  confidence >= self.confidence_threshold):
                return True
            elif (total_detected >= self.min_keypoints + 2 and confidence >= self.confidence_threshold + 0.1):
                return True
        
        return False
    
    def process_batch(self, image_paths: List[str]) -> Dict[str, dict]:
        # Pre-load all images in parallel
        images = []
        valid_paths = []
        
        start_time = time.time()
        
        for img_path in image_paths:
            img = self.preprocess_image(img_path)
            if img is not None:
                images.append(img)
                valid_paths.append(img_path)
        
        load_time = time.time() - start_time
        self.logger.info(f"Loaded {len(valid_paths)} images in {load_time:.2f}s")
        
        results = {}
        batch_start_time = time.time()
        
        # Use CUDA events for more accurate GPU timing
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
        
        # Process all images one by one
        for i, (img_path, img) in enumerate(zip(valid_paths, images)):
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
        
        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            gpu_time = start_event.elapsed_time(end_event) / 1000.0
            self.logger.info(f"GPU processing time: {gpu_time:.2f}s")
        
        # Final FPS calculation for the whole batch
        total_time = time.time() - batch_start_time
        avg_fps = len(valid_paths) / total_time if total_time > 0 else 0
        
        self.logger.info(f"Batch complete: {len(valid_paths)} images processed in {total_time:.2f}s - FPS: {avg_fps:.2f}")
        
        return results
    
    def process_image_list(self, image_paths: List[str], output_dir: str = None):
        """Process a list of images in batches and optionally save results"""
        num_images = len(image_paths)
        self.logger.info(f"Starting inference on {num_images} images")
        
        results = {}
        pedestrians_found = 0
        total_pedestrians = 0
        
        # Process in batches
        for i in range(0, num_images, self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            batch_results = self.process_batch(batch_paths)
            
            results.update(batch_results)
            
            # Update statistics
            for result in batch_results.values():
                if result['is_pedestrian']:
                    pedestrians_found += 1
                total_pedestrians += result['num_pedestrians']
                
            self.logger.info(f"Progress: {min(i + self.batch_size, num_images)}/{num_images} images processed")
        
        self.logger.info(f"Inference completed. Found pedestrians in {pedestrians_found}/{num_images} images")
        self.logger.info(f"Total pedestrians detected: {total_pedestrians}")
        
        # Save results if output directory is provided
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
        
        # Group images by pedestrian detection status
        pedestrian_images = []
        non_pedestrian_images = []
        
        for img_path, result in results.items():
            if result['is_pedestrian']:
                pedestrian_images.append(os.path.basename(img_path))
            else:
                non_pedestrian_images.append(os.path.basename(img_path))
        
        # Save lists of image names
        pedestrian_path = os.path.join(output_dir, "pedestrian_images.txt")
        with open(pedestrian_path, 'w') as f:
            f.write('\n'.join(pedestrian_images))
        
        non_pedestrian_path = os.path.join(output_dir, "non_pedestrian_images.txt")
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
        
        # Also save a summary file
        summary_path = os.path.join(output_dir, "summary.json")
        summary = {
            "total_images": len(results),
            "images_with_pedestrians": len(pedestrian_images),
            "images_without_pedestrians": len(non_pedestrian_images),
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