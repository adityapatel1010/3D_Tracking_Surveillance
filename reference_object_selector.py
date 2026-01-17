"""
Reference Object Selector using Segment Anything Model (SAM)
Allows interactive object segmentation through point clicks.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ReferenceObject:
    """Store information about the segmented reference object."""
    mask: np.ndarray  # Binary mask (H, W)
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    centroid: Tuple[int, int]  # (x, y)
    area: int
    click_point: Tuple[int, int]  # Original click point


class ReferenceObjectSelector:
    """
    SAM-based object segmentation for reference object selection.
    Supports both full SAM and MobileSAM.
    """
    
    def __init__(self, model_type: str = "vit_h", use_mobile: bool = True):
        """
        Initialize SAM model for object segmentation.
        
        Args:
            model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
            use_mobile: Use MobileSAM for faster inference
        """
        logger.info("🎯 Loading SAM model for object segmentation...")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"   Using device: {self.device}")
        
        self.use_mobile = use_mobile
        
        try:
            if use_mobile:
                self._load_mobile_sam()
            else:
                self._load_full_sam(model_type)
                
            logger.info("✅ SAM model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load SAM model: {e}")
            raise RuntimeError(f"SAM initialization failed: {e}")
    
    def _load_mobile_sam(self):
        """Load MobileSAM for faster inference."""
        try:
            from mobile_sam import sam_model_registry, SamPredictor
            
            # Try to load MobileSAM checkpoint
            checkpoint_path = "mobile_sam.pt"
            
            # Download if not exists
            if not Path(checkpoint_path).exists():
                logger.info("📥 Downloading MobileSAM checkpoint...")
                import urllib.request
                url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
                urllib.request.urlretrieve(url, checkpoint_path)
            
            model = sam_model_registry["vit_t"](checkpoint=checkpoint_path)
            model.to(self.device)
            
            self.predictor = SamPredictor(model)
            logger.info("   Loaded MobileSAM (vit_t)")
            
        except ImportError:
            logger.warning("MobileSAM not installed, falling back to full SAM")
            self._load_full_sam("vit_b")
    
    def _load_full_sam(self, model_type: str):
        """Load full SAM model."""
        from segment_anything import sam_model_registry, SamPredictor
        
        # Map model types to checkpoint URLs
        checkpoint_urls = {
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        }
        
        checkpoint_path = f"sam_{model_type}.pth"
        
        # Download if not exists
        if not Path(checkpoint_path).exists():
            logger.info(f"📥 Downloading SAM {model_type} checkpoint...")
            import urllib.request
            urllib.request.urlretrieve(checkpoint_urls[model_type], checkpoint_path)
        
        model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        model.to(self.device)
        
        self.predictor = SamPredictor(model)
        logger.info(f"   Loaded SAM ({model_type})")
    
    def set_image(self, image: np.ndarray):
        """
        Set the image for segmentation.
        
        Args:
            image: Input image (RGB, uint8)
        """
        # Convert BGR to RGB if needed
        if image.shape[2] == 3 and len(image.shape) == 3:
            # Assume OpenCV BGR format, convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        self.predictor.set_image(image_rgb)
        self.current_image_shape = image.shape[:2]
    
    def segment_from_point(self, x: int, y: int, 
                          return_all: bool = False) -> ReferenceObject:
        """
        Segment object at the clicked point.
        
        Args:
            x: X coordinate of click
            y: Y coordinate of click
            return_all: Return all mask candidates (for debugging)
        
        Returns:
            ReferenceObject with segmentation information
        """
        # Create point prompt
        point_coords = np.array([[x, y]])
        point_labels = np.array([1])  # 1 = foreground point
        
        # Predict masks
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,  # Get multiple mask candidates
        )
        
        # Select best mask (highest score)
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
        
        # Calculate properties
        bbox = self._mask_to_bbox(mask)
        centroid = self._calculate_centroid(mask)
        area = int(np.sum(mask))
        
        ref_obj = ReferenceObject(
            mask=mask.astype(np.uint8),
            bbox=bbox,
            centroid=centroid,
            area=area,
            click_point=(x, y)
        )
        
        logger.info(f"   Segmented object: area={area}, bbox={bbox}, centroid={centroid}")
        
        if return_all:
            return ref_obj, masks, scores
        
        return ref_obj
    
    def segment_from_box(self, x1: int, y1: int, x2: int, y2: int) -> ReferenceObject:
        """
        Segment object using bounding box prompt.
        
        Args:
            x1, y1: Top-left corner
            x2, y2: Bottom-right corner
        
        Returns:
            ReferenceObject with segmentation information
        """
        # Create box prompt
        box = np.array([x1, y1, x2, y2])
        
        # Predict mask
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None, :],
            multimask_output=False,
        )
        
        mask = masks[0]
        
        # Calculate properties
        bbox = self._mask_to_bbox(mask)
        centroid = self._calculate_centroid(mask)
        area = int(np.sum(mask))
        
        ref_obj = ReferenceObject(
            mask=mask.astype(np.uint8),
            bbox=bbox,
            centroid=centroid,
            area=area,
            click_point=centroid  # Use centroid as click point
        )
        
        return ref_obj
    
    def _mask_to_bbox(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Convert binary mask to bounding box."""
        rows, cols = np.where(mask)
        
        if len(rows) == 0:
            return (0, 0, 0, 0)
        
        y1, y2 = rows.min(), rows.max()
        x1, x2 = cols.min(), cols.max()
        
        return (int(x1), int(y1), int(x2), int(y2))
    
    def _calculate_centroid(self, mask: np.ndarray) -> Tuple[int, int]:
        """Calculate centroid of binary mask."""
        rows, cols = np.where(mask)
        
        if len(rows) == 0:
            return (0, 0)
        
        centroid_y = int(np.mean(rows))
        centroid_x = int(np.mean(cols))
        
        return (centroid_x, centroid_y)
    
    @staticmethod
    def visualize_mask(image: np.ndarray, mask: np.ndarray, 
                      alpha: float = 0.5, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Overlay mask on image with transparency.
        
        Args:
            image: Original image (BGR)
            mask: Binary mask
            alpha: Transparency (0-1)
            color: Mask color (BGR)
        
        Returns:
            Image with mask overlay
        """
        overlay = image.copy()
        
        # Create colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = color
        
        # Blend
        result = cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0)
        
        return result
    
    @staticmethod
    def draw_mask_contour(image: np.ndarray, mask: np.ndarray, 
                         color: Tuple[int, int, int] = (0, 255, 0),
                         thickness: int = 2) -> np.ndarray:
        """
        Draw contour around mask.
        
        Args:
            image: Original image (BGR)
            mask: Binary mask
            color: Contour color (BGR)
            thickness: Line thickness
        
        Returns:
            Image with contour drawn
        """
        result = image.copy()
        
        # Find contours
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Draw contours
        cv2.drawContours(result, contours, -1, color, thickness)
        
        return result
