"""
Reference Object-Based Tap Detection System
Integrates RT-DETR + SmolVLM + Depth Estimation + SAM for distance-based tracking.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Callable
import torch
import logging
from ultralytics import RTDETR

from depth_estimator import DepthEstimator
from distance_tracker import DistanceBasedTracker, PersonTrackingInfo
from reference_object_selector import ReferenceObject
from tap_detector import SmolVLMTapDetector, EventLogger

logger = logging.getLogger(__name__)


class ReferenceBasedTapDetector:
    """
    Main tracking class that uses reference object and depth for intelligent person tracking.
    """
    
    def __init__(self,
                 rtdetr_model: str = "rtdetr-x.pt",
                 conf_threshold: float = 0.7,
                 distance_threshold: float = 0.3):
        """
        Initialize the reference-based tap detector.
        
        Args:
            rtdetr_model: Path to RT-DETR model
            conf_threshold: Detection confidence threshold
            distance_threshold: Max distance from reference object
        """
        logger.info("🚀 Initializing Reference-Based Tap Detection System")
        
        # Load RT-DETR for person detection
        logger.info(f"📦 Loading RT-DETR model: {rtdetr_model}")
        self.detector = RTDETR(rtdetr_model)
        self.conf_threshold = conf_threshold
        
        # VLM for tap detection
        logger.info("📦 Loading SmolVLM...")
        self.vlm = SmolVLMTapDetector()
        
        # Will be initialized when processing starts
        self.dist_tracker: Optional[DistanceBasedTracker] = None
        self.depth_estimator: Optional[DepthEstimator] = None
        self.event_logger: Optional[EventLogger] = None
        
        self.distance_threshold = distance_threshold
        
        logger.info("✅ Reference-Based Tap Detector initialized")
    
    def process_video(self,
                     video_path: str,
                     ref_object: ReferenceObject,
                     depth_estimator: DepthEstimator,
                     check_interval: int = 30,
                     initial_frame: int = 0,
                     broadcast_callback: Optional[Callable] = None):
        """
        Process video with reference object-based tracking.
        
        Args:
            video_path: Path to input video
            ref_object: Reference object from SAM
            depth_estimator: Depth estimator instance
            check_interval: Frames between VLM checks
            initial_frame: Starting frame number
            broadcast_callback: Callback for real-time updates
        
        Returns:
            Tuple of (detections_dict, tracked_people_dict)
        """
        logger.info(f"🎬 Processing video: {video_path}")
        logger.info(f"   Reference object at: {ref_object.centroid}")
        logger.info(f"   Check interval: {check_interval} frames")
        
        # Initialize depth estimator and distance tracker
        self.depth_estimator = depth_estimator
        self.dist_tracker = DistanceBasedTracker(
            ref_object_centroid=ref_object.centroid,
            depth_estimator=depth_estimator,
            distance_threshold=self.distance_threshold
        )
        
        # Initialize event logger
        video_filename = Path(video_path).name
        self.event_logger = EventLogger(video_filename, broadcast_callback)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"   Video: {total_frames} frames @ {fps:.2f} fps")
        
        # Storage for visualizations
        video_detections = {}  # frame_number -> detection_data
        frames_to_check = []  # Frames accumulated for VLM check
        
        frame_number = initial_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, initial_frame)
        
        current_active_person = None
        frames_since_last_check = 0
        
        logger.info("🎯 Starting frame-by-frame processing...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect all people in frame with improved tracker parameters
            results = self.detector.track(
                frame,
                persist=True,
                conf=self.conf_threshold,
                classes=[0],  # person class
                verbose=False,
                tracker="bytetrack.yaml",  # Use ByteTrack for better tracking
                iou=0.5,  # Higher IOU threshold for more persistent tracking
                max_age=60  # Keep tracks alive for 60 frames without detection
            )
            
            # Extract detections
            detections = []
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    if boxes.id is not None:
                        track_id = int(boxes.id[i])
                        bbox = boxes.xyxy[i].cpu().numpy()
                        detections.append({
                            'track_id': track_id,
                            'bbox': bbox.tolist()
                        })
            
            # Update distance tracker
            active_person = self.dist_tracker.update(frame, detections, frame_number)
            
            # Debug logging - show all people and their status
            if frame_number % 30 == 0:  # Every 30 frames
                logger.info(f"📊 Frame {frame_number} - People status:")
                for pid, person in self.dist_tracker.tracked_people.items():
                    status = "✅ ACTIVE" if active_person and pid == active_person.track_id else "⬜ inactive"
                    approaching = "→" if person.is_approaching else "↔"
                    logger.info(f"   ID {pid} ({person.color_name}): {person.current_distance:.3f}m {approaching} {status}")
            
            # Check if active person changed
            if active_person != current_active_person:
                current_active_person = active_person
                
                if active_person:
                    # New person is now active - log it
                    self.event_logger.log_new_person(
                        track_id=active_person.track_id,
                        color_name=active_person.color_name,
                        frame_number=frame_number,
                        is_initial=(active_person.frame_count == 1),
                        color_rgb=active_person.color
                    )
            
            # Store detection data for visualization
            video_detections[frame_number] = {
                'all_detections': detections,
                'active_person': active_person.track_id if active_person else None,
                'active_bbox': active_person.bbox if active_person else None,
                'active_distance': active_person.current_distance if active_person else None
            }
            
            # Check if we should analyze with VLM
            if active_person and not active_person.has_tapped:
                frames_to_check.append(frame.copy())
                frames_since_last_check += 1
                
                # Time to check with VLM
                if frames_since_last_check >= check_interval:
                    logger.info(f"🔍 Analyzing frames {frame_number - check_interval} to {frame_number}")
                    
                    # Create annotated frame with ALL people's bounding boxes
                    annotated_frame = frame.copy()
                    
                    # Draw all tracked people with their unique colors
                    for person_id, person in self.dist_tracker.tracked_people.items():
                        if person.frames_since_last_seen == 0:  # Only visible people
                            x1, y1, x2, y2 = map(int, person.bbox)
                            
                            # Draw bounding box
                            thickness = 5 if person_id == active_person.track_id else 2
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), 
                                        person.color, thickness)
                            
                            # Add label
                            label = person.color_name
                            if person_id == active_person.track_id:
                                label += " (ACTIVE)"
                            
                            cv2.putText(annotated_frame, label,
                                      (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                      0.6, person.color, 2)
                    
                    # Ask VLM if person tapped
                    tap_result = self._check_single_person_tap(
                        annotated_frame,
                        active_person
                    )
                    
                    if tap_result:
                        self.dist_tracker.mark_tap(active_person.track_id, frame_number)
                        self.event_logger.log_tap_event(
                            track_id=active_person.track_id,
                            color_name=active_person.color_name,
                            frame_number=frame_number
                        )
                    
                    # Reset accumulation
                    frames_to_check = []
                    frames_since_last_check = 0
            
            # Progress update
            if frame_number % 100 == 0:
                progress = int((frame_number / total_frames) * 100)
                if broadcast_callback:
                    broadcast_callback({
                        'status': 'processing',
                        'progress': progress,
                        'message': f'Processing frame {frame_number}/{total_frames}'
                    })
            
            frame_number += 1
        
        cap.release()
        
        # Log summary - pass the actual PersonTrackingInfo objects
        self.event_logger.log_summary(
            tracked_people=self.dist_tracker.tracked_people,
            total_frames=frame_number - initial_frame
        )
        
        logger.info(f"✅ Processing complete!")
        summary = self.dist_tracker.get_summary()
        logger.info(f"   Total people: {summary['total_people']}")
        logger.info(f"   Tapped: {summary['people_tapped']}")
        
        return video_detections, self.dist_tracker.tracked_people
    
    def _check_single_person_tap(self, frame: np.ndarray, 
                                 person: PersonTrackingInfo) -> bool:
        """
        Check if a single person tapped using VLM.
        
        Args:
            frame: Annotated frame with person bounding box
            person: Person tracking info
        
        Returns:
            True if person tapped, False otherwise
        """
        # Simplified prompt for single person
        prompt = f"""You are analyzing a payment terminal surveillance video.
        
The person in the {person.color_name} bounding box is approaching the payment terminal.

Did this person tap their payment card on the terminal?

Answer with ONLY "YES" or "NO"."""
        
        # Query VLM
        try:
            response = self.vlm.detect_tap_multi_frame(
                frames=[frame],
                person_colors={person.track_id: person.color_name},
                event_logger=self.event_logger,
                check_number=0,
                frame_numbers=[0]
            )
            
            # Parse response
            if person.track_id in response and response[person.track_id].get('tapped'):
                return True
            
        except Exception as e:
            logger.error(f"VLM check failed: {e}")
        
        return False


def visualize_reference_tracking(video_path: str,
                                 video_detections: Dict,
                                 tracked_people: Dict,
                                 ref_object: ReferenceObject,
                                 output_path: str,
                                 initial_frame: int = 0):
    """
    Create visualization video showing reference object and active person tracking.
    
    Args:
        video_path: Input video path
        video_detections: Detection data from processing
        tracked_people: Tracked people dict
        ref_object: Reference object
        output_path: Output video path
        initial_frame: Starting frame
    """
    logger.info(f"🎨 Creating visualization video: {output_path}")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, initial_frame)
    frame_number = initial_frame
    
    # Counters for logging
    frames_with_active = 0
    boxes_drawn_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_number in video_detections:
            det = video_detections[frame_number]
            
            # Draw reference object
            ref_x, ref_y = ref_object.centroid
            cv2.circle(frame, (ref_x, ref_y), 15, (0, 0, 255), -1)
            cv2.putText(frame, "REFERENCE", (ref_x - 40, ref_y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Draw ONLY the active person with their colored bounding box
            if det['active_person'] is not None:
                frames_with_active += 1
                person_id = det['active_person']
                if person_id in tracked_people:
                    person = tracked_people[person_id]
                    x1, y1, x2, y2 = map(int, det['active_bbox'])
                    
                    # Draw colored bounding box (thick)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), person.color, 4)
                    boxes_drawn_count += 1
                    
                    # Create label with distance
                    label = f"{person.color_name} - {det['active_distance']:.3f}m"
                    if person.has_tapped:
                        label += " [TAPPED]"
                    
                    # Draw label with person's color
                    cv2.putText(frame, label, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, person.color, 2)
                    
                    # Draw line from person to reference
                    person_cx = (x1 + x2) // 2
                    person_cy = (y1 + y2) // 2
                    cv2.line(frame, (person_cx, person_cy), (ref_x, ref_y),
                           person.color, 2)
        
        out.write(frame)
        frame_number += 1
    
    cap.release()
    out.release()
    
    logger.info(f"✅ Visualization complete: {output_path}")
    logger.info(f"   Frames with active person: {frames_with_active}/{frame_number - initial_frame}")
    logger.info(f"   Frames with bounding boxes drawn: {boxes_drawn_count}")
