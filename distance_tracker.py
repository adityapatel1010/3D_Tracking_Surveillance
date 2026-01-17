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
            # Consider approaching if: distance is decreasing OR already very close
            self.is_approaching = (decreasing_count >= 1) or (distance < 0.2)
        else:
            # Not enough history, assume approaching if reasonably close
            self.is_approaching = distance < 1.0
    
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
                 zone_radius: float = 0.5,  # Zone in METERS (50cm default)
                 max_unseen_frames: int = 30):
        """
        Initialize zone-based tracker with depth estimation.
        
        Args:
            ref_object_centroid: (x, y) centroid of reference object
            depth_estimator: DepthEstimator instance for accurate 3D distance
            zone_radius: Zone radius in METERS (default 0.5m = 50cm)
            max_unseen_frames: Max frames before removing a person
        """
        self.ref_centroid = ref_object_centroid
        self.depth_estimator = depth_estimator
        self.zone_radius = zone_radius  # In meters
        self.max_unseen_frames = max_unseen_frames
        
        self.tracked_people: Dict[int, PersonTrackingInfo] = {}
        self.current_active_person: Optional[int] = None
        
        # Color assignment
        from tap_detector import PERSON_COLORS, get_color_for_person
        self.color_palette = PERSON_COLORS
        self.get_color = get_color_for_person
        
        logger.info(f"🎯 Zone-based tracker initialized (depth-aware)")
        logger.info(f"   Reference centroid: {self.ref_centroid}")
        logger.info(f"   Zone radius: {zone_radius}m ({zone_radius*100:.0f}cm)")

    
    def update(self, 
               frame: np.ndarray,
               detections: List[dict],
               frame_number: int) -> Optional[PersonTrackingInfo]:
        """
        Update tracker with depth-based zone tracking.
        
        Args:
            frame: Current frame (BGR)
            detections: List of person detections with 'track_id' and 'bbox'
            frame_number: Current frame number
        
        Returns:
            PersonTrackingInfo of the person in the zone, or None
        """
        # Generate depth map for accurate 3D distance
        depth_map = self.depth_estimator.estimate_depth(frame)
        
        # Update all tracked people and check zone membership
        detected_ids = set()
        people_in_zone = []
        
        for detection in detections:
            track_id = detection['track_id']
            bbox = detection['bbox']  # [x1, y1, x2, y2]
            
            detected_ids.add(track_id)
            
            # Calculate person centroid
            person_centroid = (
                int((bbox[0] + bbox[2]) / 2),
                int((bbox[1] + bbox[3]) / 2)
            )
            
            # Calculate 3D distance using depth estimation
            distance_3d = self.depth_estimator.calculate_3d_distance(
                depth_map,
                self.ref_centroid,
                person_centroid,
                focal_length=1000.0
            )
            
            # Check if person is in zone (depth-based)
            in_zone = distance_3d <= self.zone_radius
            
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
            person.current_distance = distance_3d  # Store in meters
            person.last_seen_frame = frame_number
            person.frames_since_last_seen = 0
            person.frame_count += 1
            
            # Add to zone list if in zone
            if in_zone:
                people_in_zone.append((track_id, distance_3d))
        
        # Mark unseen people
        for track_id in self.tracked_people:
            if track_id not in detected_ids:
                self.tracked_people[track_id].frames_since_last_seen += 1
        
        # Remove people not seen for too long
        to_remove = [
            tid for tid, person in self.tracked_people.items()
            if person.frames_since_last_seen > self.max_unseen_frames
        ]
        for tid in to_remove:
            logger.info(f"👋 Removing person {tid} (not seen for {self.max_unseen_frames} frames)")
            del self.tracked_people[tid]
        
        # Zone-based selection logic
        active_person = None
        
        if len(people_in_zone) > 0:
            # If someone is already active and still in zone, keep them
            if self.current_active_person is not None:
                active_still_in_zone = any(tid == self.current_active_person for tid, _ in people_in_zone)
                if active_still_in_zone and self.current_active_person in self.tracked_people:
                    active_person = self.tracked_people[self.current_active_person]
                    # Less verbose logging for continuous tracking
                else:
                    # Active person left zone
                    if self.current_active_person in self.tracked_people:
                        dist = self.tracked_people[self.current_active_person].current_distance
                        logger.info(f"🚶 Person {self.current_active_person} left zone (now {dist:.2f}m = {dist*100:.0f}cm)")
                    self.current_active_person = None
            
            # If no active person, select closest one in zone
            if active_person is None:
                # Sort by distance, pick closest
                people_in_zone.sort(key=lambda x: x[1])
                closest_id = people_in_zone[0][0]
                closest_dist = people_in_zone[0][1]
                active_person = self.tracked_people[closest_id]
                self.current_active_person = closest_id
                logger.info(f"🎯 Person {closest_id} ({active_person.color_name}) entered zone: {closest_dist:.2f}m ({closest_dist*100:.0f}cm)")
        else:
            # No one in zone
            if self.current_active_person is not None:
                logger.info(f"📍 Zone empty (all outside {self.zone_radius}m)")
                self.current_active_person = None
        
        return active_person
    
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
