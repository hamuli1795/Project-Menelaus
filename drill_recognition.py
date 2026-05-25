"""
Basketball Drill Recognition & Movement Pattern Learning
Trains a model to recognize specific basketball drills and movements
"""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
import pickle
from datetime import datetime
from collections import deque
import warnings

warnings.filterwarnings("ignore")

# ─── BASKETBALL DRILL DEFINITIONS ───────────────────────────────────────────

BASKETBALL_DRILLS = {
    "crossover_dribble": {
        "description": "Ball crosses in front of body from one hand to other",
        "key_movements": ["ball_transfer", "weight_shift", "low_bounce"],
        "expected_joints": ["left_wrist", "right_wrist", "left_hip", "right_hip"],
    },
    "between_legs": {
        "description": "Dribble passes between legs while moving",
        "key_movements": ["leg_separation", "bounce_timing", "directional_change"],
        "expected_joints": ["left_knee", "right_knee", "left_ankle", "right_ankle"],
    },
    "behind_back": {
        "description": "Dribble passes behind the back",
        "key_movements": ["torso_rotation", "arm_extension", "ball_timing"],
        "expected_joints": ["left_shoulder", "right_shoulder", "left_wrist", "right_wrist"],
    },
    "figure_eight": {
        "description": "Figure-8 dribbling pattern around legs",
        "key_movements": ["circular_motion", "weight_transfer", "rhythm"],
        "expected_joints": ["left_hip", "right_hip", "left_ankle", "right_ankle"],
    },
    "speed_dribble": {
        "description": "Fast dribbling in straight line",
        "key_movements": ["high_speed", "consistent_rhythm", "forward_lean"],
        "expected_joints": ["left_knee", "right_knee", "torso"],
    },
    "stationary_dribble": {
        "description": "Dribbling in place",
        "key_movements": ["vertical_bounce", "stable_base", "low_center"],
        "expected_joints": ["left_knee", "right_knee", "left_hip", "right_hip"],
    },
}


@dataclass
class MotionSignature:
    """Stores temporal motion pattern for a drill"""
    drill_name: str
    frames_data: List[np.ndarray]  # List of keypoint arrays
    timestamps: List[float]
    joint_angles: Dict[str, List[float]]
    wrist_velocity: List[float]
    hip_sway: List[float]
    confidence_scores: List[float]
    
    def save(self, path: str):
        """Save signature to file"""
        data = {
            "drill_name": self.drill_name,
            "frames_data": [f.tolist() for f in self.frames_data],
            "timestamps": self.timestamps,
            "joint_angles": self.joint_angles,
            "wrist_velocity": self.wrist_velocity,
            "hip_sway": self.hip_sway,
            "confidence_scores": self.confidence_scores,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def load(path: str) -> "MotionSignature":
        """Load signature from file"""
        with open(path, "r") as f:
            data = json.load(f)
        
        return MotionSignature(
            drill_name=data["drill_name"],
            frames_data=[np.array(f) for f in data["frames_data"]],
            timestamps=data["timestamps"],
            joint_angles=data["joint_angles"],
            wrist_velocity=data["wrist_velocity"],
            hip_sway=data["hip_sway"],
            confidence_scores=data["confidence_scores"],
        )


class DrillRecognizer:
    """Recognizes basketball drills from pose data"""
    
    KEYPOINT_MAP = {
        "NOSE": 0,
        "LEFT_EYE": 1,
        "RIGHT_EYE": 2,
        "LEFT_SHOULDER": 11,
        "RIGHT_SHOULDER": 12,
        "LEFT_ELBOW": 13,
        "RIGHT_ELBOW": 14,
        "LEFT_WRIST": 15,
        "RIGHT_WRIST": 16,
        "LEFT_HIP": 23,
        "RIGHT_HIP": 24,
        "LEFT_KNEE": 25,
        "RIGHT_KNEE": 26,
        "LEFT_ANKLE": 27,
        "RIGHT_ANKLE": 28,
    }
    
    def __init__(self, model_dir: str = "./drill_models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.7,
        )
        
        self.trained_signatures: Dict[str, MotionSignature] = {}
        self._load_trained_models()
    
    def _load_trained_models(self):
        """Load all trained drill signatures"""
        for sig_file in self.model_dir.glob("*.json"):
            try:
                sig = MotionSignature.load(str(sig_file))
                self.trained_signatures[sig.drill_name] = sig
                print(f"✓ Loaded trained model: {sig.drill_name}")
            except Exception as e:
                print(f"✗ Failed to load {sig_file}: {e}")
    
    @staticmethod
    def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle at p2 formed by p1-p2-p3"""
        v1 = p1[:2] - p2[:2]
        v2 = p3[:2] - p2[:2]
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        
        return np.degrees(angle)
    
    @staticmethod
    def calculate_velocity(p_prev: np.ndarray, p_curr: np.ndarray) -> float:
        """Calculate velocity between two points"""
        return np.linalg.norm(p_curr[:2] - p_prev[:2])
    
    def extract_features(self, keypoints: np.ndarray) -> Dict:
        """Extract movement features from keypoints"""
        features = {}
        
        # Joint angles
        features["left_elbow"] = self.calculate_angle(
            keypoints[self.KEYPOINT_MAP["LEFT_SHOULDER"]],
            keypoints[self.KEYPOINT_MAP["LEFT_ELBOW"]],
            keypoints[self.KEYPOINT_MAP["LEFT_WRIST"]],
        )
        features["right_elbow"] = self.calculate_angle(
            keypoints[self.KEYPOINT_MAP["RIGHT_SHOULDER"]],
            keypoints[self.KEYPOINT_MAP["RIGHT_ELBOW"]],
            keypoints[self.KEYPOINT_MAP["RIGHT_WRIST"]],
        )
        features["left_knee"] = self.calculate_angle(
            keypoints[self.KEYPOINT_MAP["LEFT_HIP"]],
            keypoints[self.KEYPOINT_MAP["LEFT_KNEE"]],
            keypoints[self.KEYPOINT_MAP["LEFT_ANKLE"]],
        )
        features["right_knee"] = self.calculate_angle(
            keypoints[self.KEYPOINT_MAP["RIGHT_HIP"]],
            keypoints[self.KEYPOINT_MAP["RIGHT_KNEE"]],
            keypoints[self.KEYPOINT_MAP["RIGHT_ANKLE"]],
        )
        
        # Body tilt (torso angle)
        features["body_tilt"] = self.calculate_angle(
            keypoints[self.KEYPOINT_MAP["LEFT_SHOULDER"]],
            keypoints[self.KEYPOINT_MAP["LEFT_HIP"]],
            keypoints[self.KEYPOINT_MAP["RIGHT_SHOULDER"]],
        )
        
        # Hip sway (horizontal distance between hips)
        hip_sway = abs(
            keypoints[self.KEYPOINT_MAP["LEFT_HIP"]][0]
            - keypoints[self.KEYPOINT_MAP["RIGHT_HIP"]][0]
        )
        features["hip_sway"] = hip_sway
        
        # Wrist height difference
        wrist_height_diff = abs(
            keypoints[self.KEYPOINT_MAP["LEFT_WRIST"]][1]
            - keypoints[self.KEYPOINT_MAP["RIGHT_WRIST"]][1]
        )
        features["wrist_height_diff"] = wrist_height_diff
        
        return features
    
    def train_drill(
        self,
        video_path: str,
        drill_name: str,
        num_frames: Optional[int] = None,
    ) -> MotionSignature:
        """
        Train on a basketball drill video
        
        Args:
            video_path: Path to drill video
            drill_name: Name of drill (from BASKETBALL_DRILLS keys)
            num_frames: Max frames to analyze (None = all)
            
        Returns:
            MotionSignature of the trained drill
        """
        if drill_name not in BASKETBALL_DRILLS:
            print(f"✗ Unknown drill: {drill_name}")
            print(f"Available drills: {', '.join(BASKETBALL_DRILLS.keys())}")
            return None
        
        print(f"\n🎥 Training on drill: {drill_name}")
        print(f"   Loading from: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        frames_data = []
        timestamps = []
        joint_angles = {
            "left_elbow": [],
            "right_elbow": [],
            "left_knee": [],
            "right_knee": [],
            "body_tilt": [],
        }
        wrist_velocity = []
        hip_sway = []
        confidence_scores = []
        
        prev_wrists = None
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if num_frames and frame_count >= num_frames:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                keypoints = np.array([
                    [lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark
                ])
                confidence = np.mean(keypoints[:, 2])
                
                if confidence > 0.5:
                    frames_data.append(keypoints)
                    timestamps.append(frame_count / fps)
                    confidence_scores.append(confidence)
                    
                    # Extract features
                    features = self.extract_features(keypoints)
                    
                    joint_angles["left_elbow"].append(features["left_elbow"])
                    joint_angles["right_elbow"].append(features["right_elbow"])
                    joint_angles["left_knee"].append(features["left_knee"])
                    joint_angles["right_knee"].append(features["right_knee"])
                    joint_angles["body_tilt"].append(features["body_tilt"])
                    
                    hip_sway.append(features["hip_sway"])
                    
                    # Calculate wrist velocity
                    if prev_wrists is not None:
                        left_vel = self.calculate_velocity(
                            prev_wrists[0], keypoints[self.KEYPOINT_MAP["LEFT_WRIST"]]
                        )
                        right_vel = self.calculate_velocity(
                            prev_wrists[1], keypoints[self.KEYPOINT_MAP["RIGHT_WRIST"]]
                        )
                        wrist_velocity.append((left_vel + right_vel) / 2)
                    else:
                        wrist_velocity.append(0)
                    
                    prev_wrists = [
                        keypoints[self.KEYPOINT_MAP["LEFT_WRIST"]],
                        keypoints[self.KEYPOINT_MAP["RIGHT_WRIST"]],
                    ]
            
            frame_count += 1
        
        cap.release()
        
        # Create signature
        signature = MotionSignature(
            drill_name=drill_name,
            frames_data=frames_data,
            timestamps=timestamps,
            joint_angles=joint_angles,
            wrist_velocity=wrist_velocity,
            hip_sway=hip_sway,
            confidence_scores=confidence_scores,
        )
        
        # Save model
        model_path = self.model_dir / f"{drill_name}.json"
        signature.save(str(model_path))
        
        self.trained_signatures[drill_name] = signature
        
        # Print stats
        print(f"✓ Training complete!")
        print(f"  Frames analyzed: {len(frames_data)}")
        print(f"  Avg confidence: {np.mean(confidence_scores):.2f}")
        print(f"  Avg elbow angle: {np.mean(joint_angles['left_elbow']) + np.mean(joint_angles['right_elbow']) / 2:.1f}°")
        print(f"  Avg wrist velocity: {np.mean(wrist_velocity):.4f}")
        print(f"  Model saved to: {model_path}\n")
        
        return signature
    
    def recognize_drill(
        self,
        video_path: str,
        window_size: int = 30,
    ) -> List[Tuple[str, float, Tuple[int, int]]]:
        """
        Recognize drills in a video
        
        Args:
            video_path: Path to video to analyze
            window_size: Frames to analyze per recognition window
            
        Returns:
            List of (drill_name, confidence, frame_range)
        """
        if not self.trained_signatures:
            print("✗ No trained models. Train at least one drill first.")
            return []
        
        print(f"\n🔍 Recognizing drills in: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        frame_buffer = deque(maxlen=window_size)
        recognitions = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                keypoints = np.array([
                    [lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark
                ])
                confidence = np.mean(keypoints[:, 2])
                
                if confidence > 0.5:
                    frame_buffer.append(keypoints)
            
            # Check for drill every window_size frames
            if len(frame_buffer) == window_size:
                best_match = self._match_drill(frame_buffer)
                if best_match[0]:  # If match found
                    recognitions.append((
                        best_match[0],
                        best_match[1],
                        (frame_count - window_size, frame_count),
                    ))
            
            frame_count += 1
        
        cap.release()
        
        # Print results
        if recognitions:
            print(f"✓ Found {len(recognitions)} drill segments:\n")
            for drill, conf, (start, end) in recognitions:
                start_sec = start / fps
                end_sec = end / fps
                print(f"  • {drill.upper()}: {conf:.1%} confidence (frames {start}-{end}, {start_sec:.1f}s-{end_sec:.1f}s)")
        else:
            print("✗ No drills recognized")
        
        print()
        return recognitions
    
    def _match_drill(self, frame_sequence: deque) -> Tuple[str, float]:
        """
        Match frame sequence against trained drills
        
        Returns:
            (drill_name, confidence) or (None, 0.0)
        """
        best_match = (None, 0.0)
        
        for drill_name, signature in self.trained_signatures.items():
            # Calculate features for sequence
            elbow_angles = []
            knee_angles = []
            wrist_vels = []
            
            for kpts in frame_sequence:
                features = self.extract_features(kpts)
                elbow_angles.append(features["left_elbow"] + features["right_elbow"] / 2)
                knee_angles.append(features["left_knee"] + features["right_knee"] / 2)
            
            # Compare with trained signature using DTW-inspired matching
            confidence = self._calculate_similarity(
                elbow_angles,
                knee_angles,
                signature,
            )
            
            if confidence > best_match[1]:
                best_match = (drill_name, confidence)
        
        return best_match
    
    def _calculate_similarity(
        self,
        elbow_angles: List[float],
        knee_angles: List[float],
        signature: MotionSignature,
    ) -> float:
        """
        Calculate similarity between current and trained pattern
        
        Returns:
            Confidence score 0.0-1.0
        """
        # Sample trained signature at same resolution
        sig_elbow_sample = self._resample_list(
            signature.joint_angles["left_elbow"],
            len(elbow_angles),
        )
        sig_knee_sample = self._resample_list(
            signature.joint_angles["left_knee"],
            len(knee_angles),
        )
        
        # Calculate RMS error
        elbow_error = np.sqrt(np.mean((np.array(elbow_angles) - np.array(sig_elbow_sample)) ** 2))
        knee_error = np.sqrt(np.mean((np.array(knee_angles) - np.array(sig_knee_sample)) ** 2))
        
        # Convert error to confidence (lower error = higher confidence)
        max_angle = 180
        elbow_conf = max(0, 1 - (elbow_error / max_angle))
        knee_conf = max(0, 1 - (knee_error / max_angle))
        
        return (elbow_conf + knee_conf) / 2
    
    @staticmethod
    def _resample_list(lst: List[float], target_len: int) -> List[float]:
        """Resample list to target length"""
        if len(lst) == target_len:
            return lst
        
        indices = np.linspace(0, len(lst) - 1, target_len)
        return [np.interp(i, np.arange(len(lst)), lst) for i in indices]


# ─── TRAINING & RECOGNITION EXAMPLES ───────────────────────────────────────

def train_on_drill(drill_name: str, video_path: str):
    """
    Train the model on a specific basketball drill
    
    Usage:
        python drill_recognition.py --train crossover_dribble path/to/video.mp4
    """
    recognizer = DrillRecognizer()
    
    if drill_name not in BASKETBALL_DRILLS:
        print(f"Unknown drill. Available options:")
        for name, info in BASKETBALL_DRILLS.items():
            print(f"  • {name}: {info['description']}")
        return
    
    recognizer.train_drill(video_path, drill_name)


def recognize_drills_in_video(video_path: str):
    """
    Recognize all drills in a video
    
    Usage:
        python drill_recognition.py --recognize path/to/video.mp4
    """
    recognizer = DrillRecognizer()
    recognitions = recognizer.recognize_drill(video_path)
    return recognitions


def interactive_training():
    """Interactive drill training mode"""
    recognizer = DrillRecognizer()
    
    print("\n" + "="*60)
    print("🏀 BASKETBALL DRILL TRAINER")
    print("="*60)
    print("\nAvailable drills to train:")
    for i, (name, info) in enumerate(BASKETBALL_DRILLS.items(), 1):
        print(f"  {i}. {name}")
        print(f"     {info['description']}\n")
    
    while True:
        print("\nOptions:")
        print("  1. Train on new drill video")
        print("  2. Recognize drills in video")
        print("  3. View trained models")
        print("  4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            print("\nAvailable drills:")
            drills = list(BASKETBALL_DRILLS.keys())
            for i, drill in enumerate(drills, 1):
                print(f"  {i}. {drill}")
            
            drill_idx = int(input("\nSelect drill number: ")) - 1
            drill_name = drills[drill_idx]
            
            video_path = input("Enter video path: ").strip()
            
            if Path(video_path).exists():
                recognizer.train_drill(video_path, drill_name)
            else:
                print("✗ File not found")
        
        elif choice == "2":
            video_path = input("Enter video path to recognize: ").strip()
            
            if Path(video_path).exists():
                recognizer.recognize_drill(video_path)
            else:
                print("✗ File not found")
        
        elif choice == "3":
            if recognizer.trained_signatures:
                print("\n✓ Trained models:")
                for name, sig in recognizer.trained_signatures.items():
                    print(f"  • {name}: {len(sig.frames_data)} frames")
            else:
                print("\n✗ No trained models yet")
        
        elif choice == "4":
            break


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--train" and len(sys.argv) > 3:
            train_on_drill(sys.argv[2], sys.argv[3])
        elif sys.argv[1] == "--recognize" and len(sys.argv) > 2:
            recognize_drills_in_video(sys.argv[2])
        else:
            print("Usage:")
            print("  python drill_recognition.py --train <drill_name> <video_path>")
            print("  python drill_recognition.py --recognize <video_path>")
            print("  python drill_recognition.py (for interactive mode)")
    else:
        interactive_training()
