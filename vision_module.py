"""
Computer Vision Trainer Assistant
Analyzes player movements against reference drill videos using pose detection
"""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from datetime import datetime
import json
from pathlib import Path


@dataclass
class KeypointFrame:
    """Stores pose keypoints for a single frame"""
    timestamp: float
    keypoints: np.ndarray  # Shape: (33, 3) [x, y, confidence] for MediaPipe
    confidence: float


@dataclass
class MovementAnalysis:
    """Stores analysis for a single movement deviation"""
    frame_number: int
    timestamp: float
    joint_name: str
    deviation_angle: float  # degrees
    expected_angle: float
    actual_angle: float
    severity: str  # "minor", "moderate", "severe"
    description: str


@dataclass
class TrainingSession:
    """Tracks a complete training session"""
    session_id: str
    reference_video_path: str
    start_time: datetime
    end_time: datetime = None
    deviations: List[MovementAnalysis] = field(default_factory=list)
    frame_count: int = 0
    total_accuracy: float = 0.0
    
    def to_dict(self):
        """Convert session to dictionary for JSON export"""
        return {
            "session_id": self.session_id,
            "reference_video": self.reference_video_path,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "frame_count": self.frame_count,
            "total_accuracy": round(self.total_accuracy, 2),
            "deviations": [
                {
                    "frame": dev.frame_number,
                    "timestamp": round(dev.timestamp, 2),
                    "joint": dev.joint_name,
                    "deviation_angle": round(dev.deviation_angle, 1),
                    "expected_angle": round(dev.expected_angle, 1),
                    "actual_angle": round(dev.actual_angle, 1),
                    "severity": dev.severity,
                    "description": dev.description,
                }
                for dev in self.deviations
            ],
        }


class PoseDetector:
    """Detects human pose using MediaPipe"""
    
    # MediaPipe keypoint indices
    KEYPOINTS = {
        "NOSE": 0,
        "LEFT_EYE": 1, "RIGHT_EYE": 2,
        "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
        "LEFT_ELBOW": 13, "RIGHT_ELBOW": 14,
        "LEFT_WRIST": 15, "RIGHT_WRIST": 16,
        "LEFT_HIP": 23, "RIGHT_HIP": 24,
        "LEFT_KNEE": 25, "RIGHT_KNEE": 26,
        "LEFT_ANKLE": 27, "RIGHT_ANKLE": 28,
    }
    
    def __init__(self, min_detection_confidence=0.7):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect pose in a frame
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            keypoints: (33, 3) array [x, y, confidence]
            overall_confidence: average confidence across detected keypoints
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            keypoints = np.array([
                [lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark
            ])
            overall_confidence = np.mean(keypoints[:, 2])
            return keypoints, overall_confidence
        
        return None, 0.0
    
    def draw_pose(self, frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """Draw skeleton on frame"""
        if keypoints is None:
            return frame
        
        frame_copy = frame.copy()
        h, w = frame_copy.shape[:2]
        
        # Convert normalized coords to pixel coords
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0.5:
                px, py = int(x * w), int(y * h)
                cv2.circle(frame_copy, (px, py), 5, (0, 255, 0), -1)
        
        # Draw connections (skeleton)
        connections = [
            (11, 12),  # shoulders
            (11, 13), (13, 15),  # left arm
            (12, 14), (14, 16),  # right arm
            (11, 23), (12, 24),  # torso
            (23, 25), (25, 27),  # left leg
            (24, 26), (26, 28),  # right leg
        ]
        
        for start, end in connections:
            if keypoints[start, 2] > 0.5 and keypoints[end, 2] > 0.5:
                x1, y1 = int(keypoints[start, 0] * w), int(keypoints[start, 1] * h)
                x2, y2 = int(keypoints[end, 0] * w), int(keypoints[end, 1] * h)
                cv2.line(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return frame_copy


class MovementAnalyzer:
    """Analyzes movement deviations between reference and actual poses"""
    
    def __init__(self, angle_threshold=15.0):  # degrees
        self.angle_threshold = angle_threshold
    
    @staticmethod
    def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Calculate angle at p2 formed by p1-p2-p3
        
        Args:
            p1, p2, p3: (x, y) coordinates
            
        Returns:
            angle in degrees
        """
        v1 = p1[:2] - p2[:2]
        v2 = p3[:2] - p2[:2]
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        
        return np.degrees(angle)
    
    def analyze_joint_angle(
        self,
        reference_kpts: np.ndarray,
        actual_kpts: np.ndarray,
        joint_name: str,
        frame_number: int,
        timestamp: float,
    ) -> MovementAnalysis:
        """
        Compare joint angles between reference and actual pose
        
        Args:
            reference_kpts: Reference pose keypoints
            actual_kpts: Actual pose keypoints
            joint_name: Name of joint to analyze
            frame_number: Current frame number
            timestamp: Frame timestamp in seconds
            
        Returns:
            MovementAnalysis with deviation details
        """
        
        # Define joint angle triplets (parent-joint-child)
        joint_angles = {
            "left_elbow": (11, 13, 15),  # left_shoulder-elbow-wrist
            "right_elbow": (12, 14, 16),
            "left_knee": (23, 25, 27),  # left_hip-knee-ankle
            "right_knee": (24, 26, 28),
            "left_hip": (11, 23, 25),  # left_shoulder-hip-knee
            "right_hip": (12, 24, 26),
        }
        
        if joint_name not in joint_angles:
            return None
        
        p1_idx, p2_idx, p3_idx = joint_angles[joint_name]
        
        ref_angle = self.calculate_angle(
            reference_kpts[p1_idx],
            reference_kpts[p2_idx],
            reference_kpts[p3_idx],
        )
        
        actual_angle = self.calculate_angle(
            actual_kpts[p1_idx],
            actual_kpts[p2_idx],
            actual_kpts[p3_idx],
        )
        
        deviation = abs(ref_angle - actual_angle)
        
        # Severity classification
        if deviation < self.angle_threshold:
            severity = "minor"
        elif deviation < self.angle_threshold * 2:
            severity = "moderate"
        else:
            severity = "severe"
        
        description = (
            f"{joint_name.upper()} angle deviation: expected {ref_angle:.1f}° "
            f"but got {actual_angle:.1f}° (diff: {deviation:.1f}°)"
        )
        
        return MovementAnalysis(
            frame_number=frame_number,
            timestamp=timestamp,
            joint_name=joint_name,
            deviation_angle=deviation,
            expected_angle=ref_angle,
            actual_angle=actual_angle,
            severity=severity,
            description=description,
        )


class TrainerAssistant:
    """Main trainer assistant that compares live video against reference drill"""
    
    def __init__(self, reference_video_path: str, output_dir: str = "./training_logs"):
        self.reference_video_path = reference_video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.pose_detector = PoseDetector()
        self.movement_analyzer = MovementAnalyzer()
        
        self.reference_keyframes = []
        self.current_session = None
        
        # Load reference video
        self._load_reference_video()
    
    def _load_reference_video(self):
        """Extract keyframes from reference video"""
        cap = cv2.VideoCapture(self.reference_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            keypoints, confidence = self.pose_detector.detect(frame)
            
            if keypoints is not None and confidence > 0.5:
                timestamp = frame_count / fps
                self.reference_keyframes.append(
                    KeypointFrame(
                        timestamp=timestamp,
                        keypoints=keypoints,
                        confidence=confidence,
                    )
                )
            
            frame_count += 1
        
        cap.release()
        print(f"✓ Loaded {len(self.reference_keyframes)} reference keyframes")
    
    def start_session(self) -> str:
        """Start a new training session"""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_session = TrainingSession(
            session_id=session_id,
            reference_video_path=self.reference_video_path,
            start_time=datetime.now(),
        )
        print(f"🚀 Training session started: {session_id}")
        return session_id
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame from live camera
        
        Args:
            frame: BGR image from camera
            
        Returns:
            Dictionary with analysis results
        """
        if self.current_session is None:
            raise RuntimeError("No active session. Call start_session() first.")
        
        keypoints, confidence = self.pose_detector.detect(frame)
        self.current_session.frame_count += 1
        
        if keypoints is None or confidence < 0.5:
            return {"status": "low_confidence", "confidence": confidence}
        
        # Find closest reference frame
        frame_idx = min(
            len(self.reference_keyframes) - 1,
            int(self.current_session.frame_count / 3),  # Assume ~3x slower live capture
        )
        ref_keyframe = self.reference_keyframes[frame_idx]
        
        # Analyze key joints
        joints_to_check = ["left_elbow", "right_elbow", "left_knee", "right_knee"]
        frame_deviations = []
        
        for joint in joints_to_check:
            deviation = self.movement_analyzer.analyze_joint_angle(
                ref_keyframe.keypoints,
                keypoints,
                joint,
                self.current_session.frame_count,
                ref_keyframe.timestamp,
            )
            
            if deviation and deviation.deviation_angle > 10:  # Only track significant deviations
                frame_deviations.append(deviation)
                self.current_session.deviations.append(deviation)
        
        # Calculate frame accuracy
        frame_accuracy = 100.0 - (
            sum(d.deviation_angle for d in frame_deviations) / len(joints_to_check)
        )
        frame_accuracy = max(0, frame_accuracy)
        
        self.current_session.total_accuracy += frame_accuracy
        
        return {
            "status": "ok",
            "frame_number": self.current_session.frame_count,
            "frame_accuracy": round(frame_accuracy, 1),
            "deviations_detected": len(frame_deviations),
            "keypoints": keypoints,
        }
    
    def end_session(self) -> str:
        """End training session and generate report"""
        if self.current_session is None:
            raise RuntimeError("No active session.")
        
        self.current_session.end_time = datetime.now()
        
        # Calculate overall accuracy
        if self.current_session.frame_count > 0:
            self.current_session.total_accuracy /= self.current_session.frame_count
        
        # Generate report
        report_path = (
            self.output_dir / f"{self.current_session.session_id}_report.json"
        )
        
        with open(report_path, "w") as f:
            json.dump(self.current_session.to_dict(), f, indent=2)
        
        # Print summary
        self._print_session_summary()
        
        print(f"📊 Report saved to: {report_path}")
        
        session_id = self.current_session.session_id
        self.current_session = None
        
        return session_id
    
    def _print_session_summary(self):
        """Print training session summary"""
        session = self.current_session
        
        print("\n" + "="*60)
        print("📈 TRAINING SESSION REPORT")
        print("="*60)
        print(f"Session ID: {session.session_id}")
        print(f"Duration: {(session.end_time - session.start_time).total_seconds():.1f}s")
        print(f"Frames Analyzed: {session.frame_count}")
        print(f"Overall Accuracy: {session.total_accuracy:.1f}%")
        print(f"Total Deviations: {len(session.deviations)}")
        
        if session.deviations:
            print("\n⚠️  Top Deviations:")
            sorted_devs = sorted(
                session.deviations, key=lambda x: x.deviation_angle, reverse=True
            )[:5]
            
            for i, dev in enumerate(sorted_devs, 1):
                print(
                    f"  {i}. {dev.description} [{dev.severity.upper()}]"
                )
        
        # Breakdown by joint
        joint_deviations = {}
        for dev in session.deviations:
            if dev.joint_name not in joint_deviations:
                joint_deviations[dev.joint_name] = []
            joint_deviations[dev.joint_name].append(dev.deviation_angle)
        
        if joint_deviations:
            print("\n📍 Deviations by Joint:")
            for joint, angles in joint_deviations.items():
                avg_dev = np.mean(angles)
                print(f"  • {joint}: avg {avg_dev:.1f}° ({len(angles)} times)")
        
        print("="*60 + "\n")


# ─── EXAMPLE USAGE ───────────────────────────────────────────────────────

def run_trainer_assistant(reference_video: str, use_camera: bool = True):
    """
    Run the trainer assistant
    
    Args:
        reference_video: Path to reference drill video
        use_camera: Use webcam (True) or another video file
    """
    
    assistant = TrainerAssistant(reference_video)
    session_id = assistant.start_session()
    
    # Input source
    if use_camera:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(reference_video)
    
    frame_count = 0
    
    print("🎥 Starting live capture... (press 'q' to end session)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip for selfie view
        frame = cv2.flip(frame, 1)
        
        # Process frame
        result = assistant.process_frame(frame)
        
        # Draw pose
        if result["status"] == "ok":
            frame = assistant.pose_detector.draw_pose(frame, result["keypoints"])
            
            # Display metrics
            accuracy = result["frame_accuracy"]
            color = (0, 255, 0) if accuracy > 80 else (0, 165, 255) if accuracy > 60 else (0, 0, 255)
            
            cv2.putText(
                frame,
                f"Accuracy: {accuracy:.1f}%",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
            )
            cv2.putText(
                frame,
                f"Deviations: {result['deviations_detected']}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
        else:
            cv2.putText(
                frame,
                "Low confidence - adjust position",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        
        cv2.imshow("Trainer Assistant", frame)
        
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # End session and print report
    assistant.end_session()
    
    return session_id


if __name__ == "__main__":
    # Example: python vision_module.py <reference_video_path>
    import sys
    
    if len(sys.argv) > 1:
        ref_video = sys.argv[1]
        run_trainer_assistant(ref_video, use_camera=True)
    else:
        print("Usage: python vision_module.py <reference_video_path>")
        print("Press 'q' during capture to end session and generate report")
