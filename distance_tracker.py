"""
Distance-based Person Tracker with Reference Object
Tracks only the closest approaching person to a reference object using depth estimation.
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class PersonTrackingInfo:
    """Information about a tracked person relative to reference object."""
    track_id: int
    color: Tuple[int, int, int]
    color_name: str
    
    # Position and distance tracking
    bbox: Optional[List[float]] = None
    centroid: Tuple[int, int] = (0, 0)
    distance_history: deque = field(default_factory=lambda: deque(maxlen=10))
    
    # State tracking
    current_distance: float = float('inf')
    is_approaching: bool = False
    is_active: bool = False  # Currently has bounding box shown
    has_tapped: bool = False
    tap_frame: Optional[int] = None
    
    # Frame tracking
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    frames_since_last_seen: int = 0
    frame_count: int = 0
    
    def update_distance(self, distance: float, frame_number: int):
        """Update distance and calculate if approaching."""
        self.distance_history.append(distance)
        self.current_distance = distance
        self.last_seen_frame = frame_number
        self.frames_since_last_seen = 0
        
        # Determine if approaching (distance decreasing over recent frames)
        if len(self.distance_history) >= 3:
            recent_distances = list(self.distance_history)[-3:]
            # Check if generally decreasing
            decreasing_count = sum(
                1 for i in range(len(recent_distances) - 1)
                if recent_distances[i+1] < recent_distances[i]
            )
            self.is_approaching = decreasing_count >= 2
        else:
            # Not enough history, assume approaching if close
            self.is_approaching = distance < 0.5
    
    def increment_unseen(self):
        """Increment frames since last seen."""
        self.frames_since_last_seen += 1


class DistanceBasedTracker:
    """
    Tracks people and selects only the closest approaching person for VLM analysis.
    """
    
    def __init__(self, 
                 ref_object_centroid: Tuple[int, int],
                 depth_estimator,
                 distance_threshold: float = 0.3,
                 approach_velocity_threshold: float = 0.01,
                 max_unseen_frames: int = 30):
        """
        Initialize distance-based tracker.
        
        Args:
            ref_object_centroid: (x, y) centroid of reference object
            depth_estimator: DepthEstimator instance
            distance_threshold: Max distance to consider for VLM analysis
            approach_velocity_threshold: Min velocity to consider as approaching
            max_unseen_frames: Max frames before removing a person
        """
        self.ref_centroid = ref_object_centroid
        self.depth_estimator = depth_estimator
        self.distance_threshold = distance_threshold
        self.approach_velocity_threshold = approach_velocity_threshold
        self.max_unseen_frames = max_unseen_frames
        
        self.tracked_people: Dict[int, PersonTrackingInfo] = {}
        self.current_active_person: Optional[int] = None
        
        # Color assignment
        from tap_detector import PERSON_COLORS, get_color_for_person
        self.color_palette = PERSON_COLORS
        self.get_color = get_color_for_person
        
        logger.info(f"🎯 Distance-based tracker initialized")
        logger.info(f"   Reference centroid: {ref_centroid}")
        logger.info(f"   Distance threshold: {distance_threshold}")
    
    def update(self, 
               frame: np.ndarray,
               detections: List[dict],
               frame_number: int) -> Optional[PersonTrackingInfo]:
        """
        Update tracker with new detections and return the closest approaching person.
        
        Args:
            frame: Current frame (BGR)
            detections: List of person detections with 'track_id' and 'bbox'
            frame_number: Current frame number
        
        Returns:
            PersonTrackingInfo of the closest approaching person, or None
        """
        # Generate depth map for this frame
        depth_map = self.depth_estimator.estimate_depth(frame)
        
        # Update all tracked people
        detected_ids = set()
        
        for detection in detections:
            track_id = detection['track_id']
            bbox = detection['bbox']  # [x1, y1, x2, y2]
            
            detected_ids.add(track_id)
            
            # Calculate person centroid
            person_centroid = (
                int((bbox[0] + bbox[2]) / 2),
                int((bbox[1] + bbox[3]) / 2)
            )
            
            # Calculate 3D distance using depth
            distance = self.depth_estimator.calculate_3d_distance(
                depth_map,
                self.ref_centroid,
                person_centroid,
                focal_length=1000.0
            )
            
            # Update or create person tracking info
            if track_id not in self.tracked_people:
                color, color_name = self.get_color(track_id)
                self.tracked_people[track_id] = PersonTrackingInfo(
                    track_id=track_id,
                    color=color,
                    color_name=color_name,
                    first_seen_frame=frame_number
                )
                logger.info(f"🆕 New person detected: ID {track_id} ({color_name})")
            
            person = self.tracked_people[track_id]
            person.bbox = bbox
            person.centroid = person_centroid
            person.update_distance(distance, frame_number)
            person.frame_count += 1
        
        # Mark unseen people
        for track_id in self.tracked_people:
            if track_id not in detected_ids:
                self.tracked_people[track_id].increment_unseen()
        
        # Remove people not seen for too long
        to_remove = [
            tid for tid, person in self.tracked_people.items()
            if person.frames_since_last_seen > self.max_unseen_frames
        ]
        for tid in to_remove:
            logger.info(f"👋 Removing person {tid} (not seen for {self.max_unseen_frames} frames)")
            del self.tracked_people[tid]
        
        # Select closest approaching person
        closest_person = self._select_closest_approaching()
        
        # Update active status
        for track_id, person in self.tracked_people.items():
            person.is_active = (closest_person is not None and 
                              track_id == closest_person.track_id)
        
        # Log if active person changed
        new_active_id = closest_person.track_id if closest_person else None
        if new_active_id != self.current_active_person:
            if new_active_id is not None:
                logger.info(f"🎯 Active person changed to: ID {new_active_id} "
                          f"({closest_person.color_name}, distance={closest_person.current_distance:.3f})")
            else:
                logger.info(f"📍 No active person (all too far or moving away)")
            self.current_active_person = new_active_id
        
        return closest_person
    
    def _select_closest_approaching(self) -> Optional[PersonTrackingInfo]:
        """Select the closest approaching person for analysis."""
        candidates = []
        
        for person in self.tracked_people.values():
            # Must be currently visible
            if person.frames_since_last_seen > 0:
                continue
                
            # Must be approaching
            if not person.is_approaching:
                continue
            
            # Must be within threshold distance
            if person.current_distance > self.distance_threshold:
                continue
            
            candidates.append(person)
        
        if not candidates:
            return None
        
        # Return closest one
        closest = min(candidates, key=lambda p: p.current_distance)
        return closest
    
    def mark_tap(self, track_id: int, frame_number: int):
        """Mark that a person has tapped."""
        if track_id in self.tracked_people:
            person = self.tracked_people[track_id]
            person.has_tapped = True
            person.tap_frame = frame_number
            logger.info(f"✅ Person {track_id} ({person.color_name}) tapped at frame {frame_number}")
    
    def get_summary(self) -> Dict:
        """Get summary of tracking results."""
        return {
            'total_people': len(self.tracked_people),
            'people_tapped': sum(1 for p in self.tracked_people.values() if p.has_tapped),
            'people_not_tapped': sum(1 for p in self.tracked_people.values() if not p.has_tapped),
            'people': [
                {
                    'track_id': p.track_id,
                    'color': p.color_name,
                    'color_rgb': list(p.color),
                    'tapped': p.has_tapped,
                    'tap_frame': p.tap_frame,
                    'min_distance': min(p.distance_history) if p.distance_history else None,
                    'first_seen_frame': p.first_seen_frame,
                    'last_seen_frame': p.last_seen_frame,
                    'frame_count': p.frame_count
                }
                for p in self.tracked_people.values()
            ]
        }
