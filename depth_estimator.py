"""
Depth Estimation Module using Depth Anything V2
Provides depth map generation and 3D distance calculation for the VTA tracking system.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class DepthEstimator:
    """
    Depth estimation using Depth Anything V2 model.
    Provides depth maps and 3D distance calculations.
    """
    
    def __init__(self, model_name: str = "depth-anything/Depth-Anything-V2-Small-hf"):
        """
        Initialize depth estimation model.
        
        Args:
            model_name: HuggingFace model name for Depth Anything V2
        """
        logger.info(f"📊 Loading Depth Estimation model: {model_name}")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"   Using device: {self.device}")
        
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("✅ Depth Estimation model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load Depth Anything model: {e}")
            logger.info("💡 Falling back to MiDaS model...")
            self._load_midas_fallback()
    
    def _load_midas_fallback(self):
        """Fallback to MiDaS if Depth Anything is not available."""
        try:
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self.model.to(self.device)
            self.model.eval()
            
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = midas_transforms.small_transform
            self.use_midas = True
            
            logger.info("✅ MiDaS model loaded successfully (fallback)")
            
        except Exception as e:
            logger.error(f"❌ Failed to load MiDaS fallback: {e}")
            raise RuntimeError("Could not load any depth estimation model")
    
    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """
        Generate depth map for a given frame.
        
        Args:
            frame: Input frame (BGR or RGB, uint8)
        
        Returns:
            Depth map (normalized to 0-1, float32)
        """
        # Convert BGR to RGB if needed
        if frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        
        with torch.no_grad():
            if hasattr(self, 'use_midas') and self.use_midas:
                # MiDaS processing
                input_batch = self.transform(frame_rgb).to(self.device)
                prediction = self.model(input_batch)
                depth_map = prediction.squeeze().cpu().numpy()
            else:
                # Depth Anything processing
                inputs = self.processor(images=frame_rgb, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth
                
                # Interpolate to original size
                depth_map = torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=frame_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze().cpu().numpy()
        
        # Normalize to 0-1
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        
        return depth_map.astype(np.float32)
    
    def get_depth_at_point(self, depth_map: np.ndarray, x: int, y: int, 
                          window_size: int = 5) -> float:
        """
        Get depth value at a specific point with averaging.
        
        Args:
            depth_map: Pre-computed depth map
            x: X coordinate
            y: Y coordinate
            window_size: Window size for averaging around the point
        
        Returns:
            Average depth value around the point
        """
        h, w = depth_map.shape
        
        # Ensure point is within bounds
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        
        # Calculate window bounds
        half_window = window_size // 2
        y_min = max(0, y - half_window)
        y_max = min(h, y + half_window + 1)
        x_min = max(0, x - half_window)
        x_max = min(w, x + half_window + 1)
        
        # Average depth in window
        depth_window = depth_map[y_min:y_max, x_min:x_max]
        avg_depth = np.mean(depth_window)
        
        return float(avg_depth)
    
    def calculate_3d_distance(self, depth_map: np.ndarray, 
                             point1: Tuple[int, int], 
                             point2: Tuple[int, int],
                             focal_length: float = 1000.0) -> float:
        """
        Calculate 3D Euclidean distance between two points using depth.
        
        Args:
            depth_map: Pre-computed depth map
            point1: (x1, y1) coordinates of first point
            point2: (x2, y2) coordinates of second point
            focal_length: Camera focal length in pixels (approximate)
        
        Returns:
            3D Euclidean distance in normalized units
        """
        x1, y1 = point1
        x2, y2 = point2
        
        # Get depth values
        d1 = self.get_depth_at_point(depth_map, x1, y1)
        d2 = self.get_depth_at_point(depth_map, x2, y2)
        
        # Convert to 3D coordinates (assuming pinhole camera model)
        # Using normalized coordinates (depth is already 0-1)
        h, w = depth_map.shape
        cx, cy = w / 2, h / 2  # Assume center of image as principal point
        
        # 3D coordinates
        X1 = (x1 - cx) * d1 / focal_length
        Y1 = (y1 - cy) * d1 / focal_length
        Z1 = d1
        
        X2 = (x2 - cx) * d2 / focal_length
        Y2 = (y2 - cy) * d2 / focal_length
        Z2 = d2
        
        # Euclidean distance
        distance = np.sqrt((X2 - X1)**2 + (Y2 - Y1)**2 + (Z2 - Z1)**2)
        
        return float(distance)
    
    def calculate_pixel_distance_with_depth(self, depth_map: np.ndarray,
                                           point1: Tuple[int, int],
                                           point2: Tuple[int, int]) -> float:
        """
        Calculate depth-weighted distance between two points.
        Simpler alternative to full 3D distance.
        
        Args:
            depth_map: Pre-computed depth map
            point1: (x1, y1) coordinates of first point
            point2: (x2, y2) coordinates of second point
        
        Returns:
            Depth-weighted distance
        """
        x1, y1 = point1
        x2, y2 = point2
        
        # 2D Euclidean distance
        pixel_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Get depth difference
        d1 = self.get_depth_at_point(depth_map, x1, y1)
        d2 = self.get_depth_at_point(depth_map, x2, y2)
        depth_diff = abs(d2 - d1)
        
        # Combined distance (depth-weighted)
        weighted_distance = np.sqrt(pixel_distance**2 + (depth_diff * 1000)**2)
        
        return float(weighted_distance)
    
    def visualize_depth_map(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Convert depth map to colored visualization.
        
        Args:
            depth_map: Depth map (0-1 normalized)
        
        Returns:
            RGB image with depth visualization
        """
        # Convert to 8-bit
        depth_8bit = (depth_map * 255).astype(np.uint8)
        
        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_INFERNO)
        
        return depth_colored
