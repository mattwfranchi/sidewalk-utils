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
from ultralytics import YOLO

try:
    import nvjpeg  # NVIDIA's GPU-accelerated JPEG decoder
    HAS_NVJPEG = True
except ImportError:
    HAS_NVJPEG = False


class YOLOInferencer:
    """
    YOLO pedestrian detector for dashcam images (single GPU version)
    """
    def __init__(
        self,
        model_path: str = 'yolov8x.pt',
        device_id: int = 0,
        batch_size: int = 32,
        confidence_threshold: float = 0.4,
        class_ids: List[int] = [0],  # Default is person class in COCO
        scale_factor: float = 0.5,  # Downscale factor for faster processing
        split_processing: bool = False,  # Add this parameter
        array_id: int = 0,  # Add this parameter
    ):
        """
        Initialize the single-GPU inferencer
        
        Args:
            model_path: Path to the YOLO model
            device_id: GPU device ID to use
            batch_size: Batch size for processing
            confidence_threshold: Minimum confidence threshold for detected objects
            class_ids: List of class IDs to consider (default: [0] for person in COCO)
            scale_factor: Factor to scale down images for faster processing
            split_processing: Whether to use split processing for higher resolution detection
            array_id: SLURM array task ID for parallel processing
        """
        self.model_path = model_path
        self.device_id = device_id
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.class_ids = class_ids
        self.scale_factor = scale_factor
        self.split_processing = split_processing
        self.array_id = array_id  # Store the array ID
        
        self.setup_logging()
        self.setup_device()
        self.load_model()

        torch.backends.cudnn.benchmark = True
        self.use_amp = True  # Enable mixed precision for YOLO
        
        # Add this check
        if split_processing and scale_factor != 1.0:
            self.logger.warning(
                f"Split processing is enabled but scale_factor is {scale_factor}. "
                f"Setting scale_factor=1.0 since it's not used in split processing mode."
            )
            self.scale_factor = 1.0
        
        # Add this clear logging about processing mode
        if self.split_processing:
            self.logger.info("🔍 SPLIT PROCESSING MODE ENABLED: Images will be processed in halves for higher resolution detection")
        else:
            self.logger.info(f"Standard processing mode: Images will be scaled by factor {self.scale_factor}")

    def setup_logging(self):
        """Configure logging for the inferencer"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            stream=sys.stdout
        )
        self.logger = logging.getLogger("YOLOInferencer")
    
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
        """Load YOLO model"""
        self.logger.info(f"Loading YOLO model from {self.model_path}")
        
        # Set device
        device = 0 if torch.cuda.is_available() else 'cpu'
        
        try:
            self.model = YOLO(self.model_path)
            self.model.to(device)
            self.logger.info("Model loaded successfully")
            self.warmup()
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def warmup(self):
        """Warmup the model with a dummy batch to optimize performance."""
        self.logger.info("Warming up model with a dummy batch...")
        
        # Create a dummy batch
        batch_size = min(4, self.batch_size)  # Small batch for warmup
        dummy_tensor = torch.zeros((batch_size, 3, 640, 640), dtype=torch.float32)
        
        if torch.cuda.is_available():
            dummy_tensor = dummy_tensor.cuda()
        
        # Run a forward pass to warm up the model
        with torch.no_grad():
            try:
                _ = self.model(dummy_tensor, verbose=False)
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
    
    def is_pedestrian(self, results) -> bool:
        """Check if detections contain pedestrians based on class and confidence"""
        if results is None or len(results) == 0:
            return False
            
        # Access detections from results
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return False
            
        # Get class IDs and confidence scores
        cls = boxes.cls.cpu().numpy() if hasattr(boxes, 'cls') else []
        conf = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else []
        
        # Check if any detected object is a pedestrian with sufficient confidence
        for i in range(len(cls)):
            if int(cls[i]) in self.class_ids and conf[i] >= self.confidence_threshold:
                return True
                
        return False
    
    def process_batch(self, image_paths: List[str]) -> Dict[str, dict]:
        if self.split_processing:
            # Add more informative logging
            self.logger.info(f"🔍 Processing batch of {len(image_paths)} images using SPLIT MODE")
            
            # Process each image individually with split processing
            results = {}
            for img_path in image_paths:
                result = self.process_split_image(img_path)
                results[img_path] = result
                
            # Add summary logging after split processing completes
            pedestrian_count = sum(1 for r in results.values() if r["is_pedestrian"])
            total_pedestrians = sum(r["num_pedestrians"] for r in results.values())
            self.logger.info(f"🔍 Split processing complete: {pedestrian_count}/{len(results)} images with pedestrians, {total_pedestrians} total pedestrians")
            
            return results
        else:
            # Pre-load all images in parallel
            images, valid_paths = self.preprocess_batch(image_paths)
            
            start_time = time.time()
            
            load_time = time.time() - start_time
            self.logger.info(f"Loaded {len(valid_paths)} images in {load_time:.2f}s")
            
            results = {}
            batch_start_time = time.time()
            
            # Add this after preprocessing in process_batch:
            if not images or len(images) == 0:
                self.logger.warning(f"No valid images to process in batch of {len(image_paths)} images")
                return {}

            # Check image dimensions
            self.logger.info(f"Processing batch: {len(images)} images, first image shape: {images[0].shape if images else 'None'}")

            # Process in true batches
            if len(valid_paths) > 0:
                try:
                    # Add debug logging
                    self.logger.info("Starting batch processing")
                    
                    # Process all images in the batch
                    batch_results = self.model.predict(
                        images, 
                        imgsz=640,
                        conf=self.confidence_threshold,
                        iou=0.45,
                        max_det=50,
                        verbose=False
                    )
                    
                    self.logger.info(f"Batch processing complete. Processed {len(batch_results)} images")
                    
                    # Process the results for each image
                    for i, (img_path, img_result) in enumerate(zip(valid_paths, batch_results)):
                        boxes = img_result.boxes
                        
                        # Convert detections to a similar format as OpenPose for consistency
                        has_pedestrian = self.is_pedestrian(img_result)
                        
                        # Count pedestrians (only count objects of target class with sufficient confidence)
                        cls = boxes.cls.cpu().numpy() if hasattr(boxes, 'cls') else []
                        conf = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else []
                        
                        # Filter person class and confidence threshold
                        num_pedestrians = sum(1 for c, cf in zip(cls, conf) if int(c) in self.class_ids and cf >= self.confidence_threshold)
                        
                        # Get bounding boxes (convert to original scale if needed)
                        if hasattr(boxes, 'xyxy'):
                            bboxes = boxes.xyxy.cpu().numpy()
                            if self.scale_factor != 1.0:
                                bboxes /= self.scale_factor
                        else:
                            bboxes = np.array([])
                        
                        # Get confidence scores
                        confidences = conf if len(conf) > 0 else np.array([])
                        
                        # Get class ids
                        class_ids = cls if len(cls) > 0 else np.array([])
                        
                        results[img_path] = {
                            "bboxes": bboxes,
                            "confidences": confidences,
                            "class_ids": class_ids,
                            "is_pedestrian": has_pedestrian,
                            "num_pedestrians": num_pedestrians
                        }
                    
                    # Add this to process_batch after model inference:
                    # Log basic detection statistics for the batch
                    pedestrian_counts = [sum(1 for c, cf in zip(boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()) 
                                             if int(c) in self.class_ids and cf >= self.confidence_threshold) 
                                         for boxes in [r.boxes for r in batch_results if hasattr(r, 'boxes')]]
                         
                    total_peds = sum(pedestrian_counts)
                    images_with_peds = sum(1 for count in pedestrian_counts if count > 0)

                    self.logger.info(f"Batch detection stats: {images_with_peds}/{len(batch_results)} images with pedestrians, {total_peds} total pedestrians")
                    
                except Exception as e:
                    self.logger.error(f"Error in batch processing: {e}")
                    # Fall back to one-by-one processing
                    for i, (img_path, img) in enumerate(zip(valid_paths, images)):
                        try:
                            result = self.model(img, imgsz=640, verbose=False)[0]
                            
                            has_pedestrian = self.is_pedestrian(result)
                            
                            # Extract detection info
                            boxes = result.boxes
                            cls = boxes.cls.cpu().numpy() if hasattr(boxes, 'cls') else []
                            conf = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else []
                            
                            # Count pedestrians
                            num_pedestrians = sum(1 for c, cf in zip(cls, conf) if int(c) in self.class_ids and cf >= self.confidence_threshold)
                            
                            # Get bounding boxes
                            if hasattr(boxes, 'xyxy'):
                                bboxes = boxes.xyxy.cpu().numpy()
                                if self.scale_factor != 1.0:
                                    bboxes /= self.scale_factor
                            else:
                                bboxes = np.array([])
                            
                            results[img_path] = {
                                "bboxes": bboxes,
                                "confidences": conf if len(conf) > 0 else np.array([]),
                                "class_ids": cls if len(cls) > 0 else np.array([]),
                                "is_pedestrian": has_pedestrian,
                                "num_pedestrians": num_pedestrians
                            }
                        except Exception as e:
                            self.logger.error(f"Error processing image {img_path}: {e}")
                            results[img_path] = {
                                "bboxes": np.array([]),
                                "confidences": np.array([]),
                                "class_ids": np.array([]),
                                "is_pedestrian": False,
                                "num_pedestrians": 0
                            }
            
            total_time = time.time() - batch_start_time
            avg_fps = len(valid_paths) / total_time if total_time > 0 else 0
            
            self.logger.info(f"Batch complete: {len(valid_paths)} images processed in {total_time:.2f}s - FPS: {avg_fps:.2f}")
            
            return results
    
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
                checkpoint_file = os.path.join(
                    checkpoint_dir, 
                    f"checkpoint_task{self.array_id}_{i+self.batch_size}.json"
                )
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
        pedestrian_path = os.path.join(output_dir, f"pedestrian_images_task{self.array_id}.txt")
        non_pedestrian_path = os.path.join(output_dir, f"non_pedestrian_images_task{self.array_id}.txt")
        summary_path = os.path.join(output_dir, f"summary_task{self.array_id}.json")
        
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
        
        # Save detailed results (bounding boxes) for images with pedestrians
        import pickle
        import json
        
        # Create task-specific subdirectory for detailed results
        details_dir = os.path.join(output_dir, f"detections_task{self.array_id}")
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
                    "bboxes": result["bboxes"].tolist() if isinstance(result["bboxes"], np.ndarray) else [],
                    "confidences": result["confidences"].tolist() if isinstance(result["confidences"], np.ndarray) else [],
                    "class_ids": result["class_ids"].tolist() if isinstance(result["class_ids"], np.ndarray) else []
                }
                
                # Save as JSON
                json_path = os.path.join(details_dir, f"{base_name}.json")
                with open(json_path, 'w') as f:
                    json.dump(result_json, f)
                
                saved_count += 1
        
        # Also save a summary file with aggregate statistics
        summary_path = os.path.join(output_dir, f"summary_task{self.array_id}.json")
        
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
        
        # Add more detailed statistics about pedestrian counts
        pedestrian_counts = [result["num_pedestrians"] for result in results.values() if result["is_pedestrian"]]
        total_pedestrians_found = sum(pedestrian_counts)
        max_pedestrians = max(pedestrian_counts) if pedestrian_counts else 0
        avg_pedestrians = sum(pedestrian_counts)/len(pedestrian_counts) if pedestrian_counts else 0

        # Add these fields to the summary dictionary:
        summary = {
            "total_images": total_images,
            "images_with_pedestrians": images_with_pedestrians,
            "images_without_pedestrians": images_without_pedestrians,
            "total_pedestrians_found": total_pedestrians_found,
            "max_pedestrians_per_image": max_pedestrians,
            "avg_pedestrians_per_image": avg_pedestrians,
            "pedestrian_count_distribution": {str(count): pedestrian_counts.count(count) for count in set(pedestrian_counts)},
            "detection_parameters": {
                "confidence_threshold": self.confidence_threshold,
                "class_ids": self.class_ids,
                "scale_factor": self.scale_factor,
                "split_processing": self.split_processing,  # Add this field
                "processing_mode": "split_image" if self.split_processing else "standard"  # Add descriptive mode
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Saved results: {len(pedestrian_images)} images with pedestrians, {len(non_pedestrian_images)} without")
        self.logger.info(f"Saved detailed detections for {saved_count} images with pedestrians")
    
    def optimize_memory_usage(self):
        """Configure for optimal memory usage"""
        if torch.cuda.is_available():
            # Enable memory caching for faster allocation
            torch.cuda.empty_cache()
            
            # Optimize memory allocation strategy
            torch.cuda.set_per_process_memory_fraction(0.97)  # Use more of available memory
            
            # Enable TF32 on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Set benchmark mode for faster convolutions
            torch.backends.cudnn.benchmark = True
            
            # Disable gradient calculation (we're only doing inference)
            torch.set_grad_enabled(False)

    def process_split_image(self, image_path: str) -> Dict:
        """Process image by splitting it into top and bottom halves for higher resolution detection"""
        try:
            # Load original image
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                return {"is_pedestrian": False, "num_pedestrians": 0, "bboxes": np.array([])}
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get dimensions
            h, w, _ = img.shape
            
            # Add detailed dimension logging
            self.logger.debug(f"🔍 Split processing image {os.path.basename(image_path)}: original dimensions {w}x{h}")
            
            # Split into top and bottom halves with overlap
            overlap = 100  # pixels of overlap to handle objects at the boundary
            top_half = img[0:h//2+overlap, :, :]
            bottom_half = img[max(0, h//2-overlap):h, :, :]
            
            # Add size logging for the halves
            self.logger.debug(f"🔍 Top half: {top_half.shape}, Bottom half: {bottom_half.shape}, Overlap: {overlap}px")
            
            # Process each half
            top_results = self.model.predict(top_half, imgsz=640, conf=self.confidence_threshold, 
                                              iou=0.45, max_det=50, verbose=False)[0]
            bottom_results = self.model.predict(bottom_half, imgsz=640, conf=self.confidence_threshold, 
                                                 iou=0.45, max_det=50, verbose=False)[0]
            
            # Extract detections from top half
            top_boxes = top_results.boxes
            top_cls = top_boxes.cls.cpu().numpy() if hasattr(top_boxes, 'cls') else []
            top_conf = top_boxes.conf.cpu().numpy() if hasattr(top_boxes, 'conf') else []
            top_bboxes = top_boxes.xyxy.cpu().numpy() if hasattr(top_boxes, 'xyxy') else np.array([])
            
            # Extract detections from bottom half (adjust y-coordinates)
            bottom_boxes = bottom_results.boxes
            bottom_cls = bottom_boxes.cls.cpu().numpy() if hasattr(bottom_boxes, 'cls') else []
            bottom_conf = bottom_boxes.conf.cpu().numpy() if hasattr(bottom_boxes, 'conf') else []
            bottom_bboxes = bottom_boxes.xyxy.cpu().numpy() if hasattr(bottom_boxes, 'xyxy') else np.array([])
            
            # Adjust bottom half coordinates
            if len(bottom_bboxes) > 0:
                # Adjust y-coordinates by adding the offset (accounting for overlap)
                bottom_offset = max(0, h//2-overlap)
                bottom_bboxes[:, 1] += bottom_offset  # y1
                bottom_bboxes[:, 3] += bottom_offset  # y2
            
            # Combine results
            all_bboxes = np.vstack([top_bboxes, bottom_bboxes]) if len(top_bboxes) > 0 and len(bottom_bboxes) > 0 else \
                        (top_bboxes if len(top_bboxes) > 0 else bottom_bboxes)
            all_cls = np.concatenate([top_cls, bottom_cls]) if len(top_cls) > 0 and len(bottom_cls) > 0 else \
                     (top_cls if len(top_cls) > 0 else bottom_cls)
            all_conf = np.concatenate([top_conf, bottom_conf]) if len(top_conf) > 0 and len(bottom_conf) > 0 else \
                      (top_conf if len(top_conf) > 0 else bottom_conf)
            
            # Apply NMS to remove duplicates at the boundary
            from torchvision.ops import nms
            import torch
            
            if len(all_bboxes) > 0:
                boxes_tensor = torch.from_numpy(all_bboxes).float()
                scores_tensor = torch.from_numpy(all_conf).float()
                
                # Apply NMS
                keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold=0.45)
                
                # Filter results
                all_bboxes = all_bboxes[keep_indices.numpy()]
                all_cls = all_cls[keep_indices.numpy()]
                all_conf = all_conf[keep_indices.numpy()]
            
            # Count pedestrians
            num_pedestrians = sum(1 for c, cf in zip(all_cls, all_conf) 
                                 if int(c) in self.class_ids and cf >= self.confidence_threshold)
            
            is_pedestrian = num_pedestrians > 0
            
            # Add detailed result logging
            self.logger.debug(f"🔍 Split results: Top half: {len(top_cls)} detections, Bottom half: {len(bottom_cls)} detections")
            
            # Log the final combined results
            self.logger.debug(f"🔍 Combined results after NMS: {num_pedestrians} pedestrians detected")
            
            return {
                "is_pedestrian": is_pedestrian,
                "num_pedestrians": num_pedestrians,
                "bboxes": all_bboxes,
                "confidences": all_conf,
                "class_ids": all_cls
            }
            
        except Exception as e:
            self.logger.error(f"Error in split processing for {image_path}: {e}")
            return {"is_pedestrian": False, "num_pedestrians": 0, "bboxes": np.array([])}