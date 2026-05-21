"""
HolisticTracker
===============
Wraps MediaPipe Holistic to extract a 258-dim keypoint vector per frame.

Feature layout (258 total):
  pose       33 landmarks × 4 values (x, y, z, visibility) = 132
  left_hand  21 landmarks × 3 values (x, y, z)             =  63
  right_hand 21 landmarks × 3 values (x, y, z)             =  63
                                                  total     = 258
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np
import mediapipe as mp


class HolisticTracker:
    """MediaPipe Holistic wrapper for real-time keypoint extraction."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing  = mp.solutions.drawing_utils
        self.mp_styles   = mp.solutions.drawing_styles
        self.holistic    = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray):
        """Process a BGR frame and return raw MediaPipe results."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.holistic.process(rgb)
        rgb.flags.writeable = True
        return results

    # ------------------------------------------------------------------
    def extract_keypoints(self, results) -> np.ndarray:
        """
        Flatten detected landmarks into a (258,) float32 vector.
        Returns zeros for any landmark group not detected.
        """
        pose = (
            np.array(
                [[lm.x, lm.y, lm.z, lm.visibility]
                 for lm in results.pose_landmarks.landmark],
                dtype=np.float32,
            ).flatten()
            if results.pose_landmarks
            else np.zeros(33 * 4, dtype=np.float32)
        )
        lh = (
            np.array(
                [[lm.x, lm.y, lm.z]
                 for lm in results.left_hand_landmarks.landmark],
                dtype=np.float32,
            ).flatten()
            if results.left_hand_landmarks
            else np.zeros(21 * 3, dtype=np.float32)
        )
        rh = (
            np.array(
                [[lm.x, lm.y, lm.z]
                 for lm in results.right_hand_landmarks.landmark],
                dtype=np.float32,
            ).flatten()
            if results.right_hand_landmarks
            else np.zeros(21 * 3, dtype=np.float32)
        )
        return np.concatenate([pose, lh, rh])   # (258,)

    # ------------------------------------------------------------------
    def draw_landmarks(self, frame: np.ndarray, results) -> np.ndarray:
        """Draw pose + hand landmarks on a BGR frame in-place and return it."""
        self.mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_styles.get_default_pose_landmarks_style(),
        )
        for hand_landmarks, color_dot, color_conn in [
            (results.left_hand_landmarks,  (121, 22, 76),  (121, 44, 250)),
            (results.right_hand_landmarks, (245, 117, 66), (245, 66, 230)),
        ]:
            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=color_dot, thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=color_conn, thickness=2, circle_radius=2),
            )
        return frame

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Release MediaPipe resources."""
        self.holistic.close()
