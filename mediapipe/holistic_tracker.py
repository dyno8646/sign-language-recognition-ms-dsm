"""MediaPipe Holistic tracking for sign language landmark extraction."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

# Local package folder is named "mediapipe"; load Google's library explicitly.
_LOCAL_PKG_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _LOCAL_PKG_DIR.parent


def _import_google_mediapipe() -> Any:
    """Import the installed MediaPipe library without breaking this repo package."""
    cached = sys.modules.get("_google_mediapipe_lib")
    if cached is not None:
        return cached

    local_pkg = sys.modules.get("mediapipe")
    if local_pkg is not None and not hasattr(local_pkg, "solutions"):
        del sys.modules["mediapipe"]

    original_path = sys.path.copy()
    try:
        sys.path = [
            p
            for p in sys.path
            if Path(p).resolve() != _PROJECT_ROOT.resolve()
        ]
        mp = importlib.import_module("mediapipe")
        if not hasattr(mp, "solutions"):
            raise ImportError(
                "Installed MediaPipe has no 'solutions' API. "
                "Use: pip install mediapipe==0.10.14"
            )
        sys.modules["_google_mediapipe_lib"] = mp
        return mp
    finally:
        sys.path[:] = original_path
        if local_pkg is not None:
            sys.modules["mediapipe"] = local_pkg


_mp = _import_google_mediapipe()
_mp_drawing = _mp.solutions.drawing_utils
_mp_drawing_styles = _mp.solutions.drawing_styles
_mp_holistic = _mp.solutions.holistic

# Landmark counts (MediaPipe Holistic)
_NUM_POSE = 33
_NUM_HAND = 21
_NUM_FACE = 468

_POSE_DIM = 4  # x, y, z, visibility
_HAND_DIM = 3  # x, y, z
_FACE_DIM = 3

_POSE_FEATURES = _NUM_POSE * _POSE_DIM  # 132
_HAND_FEATURES = _NUM_HAND * _HAND_DIM  # 63
_TOTAL_KEYPOINTS = _POSE_FEATURES + _HAND_FEATURES + _HAND_FEATURES  # 258


def _landmarks_to_array(
    landmarks: Optional[Any],
    num_points: int,
    dims: int,
) -> np.ndarray:
    """Convert MediaPipe landmark list to a fixed-size numpy array."""
    out = np.zeros((num_points, dims), dtype=np.float32)
    if landmarks is None:
        return out

    for i, lm in enumerate(landmarks.landmark):
        if i >= num_points:
            break
        out[i, 0] = lm.x
        out[i, 1] = lm.y
        if dims >= 3:
            out[i, 2] = lm.z
        if dims >= 4:
            out[i, 3] = getattr(lm, "visibility", 0.0)
    return out


def _array_to_landmark_list(array: np.ndarray) -> Any:
    """Convert a numpy landmark array back to a MediaPipe NormalizedLandmarkList."""
    landmark_pb2 = _mp.framework.formats.landmark_pb2
    landmark_list = landmark_pb2.NormalizedLandmarkList()
    for row in array:
        lm = landmark_list.landmark.add()
        lm.x = float(row[0])
        lm.y = float(row[1])
        if array.shape[1] >= 3:
            lm.z = float(row[2])
        if array.shape[1] >= 4:
            lm.visibility = float(row[3])
    return landmark_list


class HolisticTracker:
    """
    Wrapper around MediaPipe Holistic for pose, hands, and face landmarks.

    Extracts fixed-size keypoint vectors for sign language recognition pipelines.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """
        Initialize the Holistic tracker.

        Args:
            min_detection_confidence: Minimum confidence for initial detection.
            min_tracking_confidence: Minimum confidence for frame-to-frame tracking.
        """
        self._holistic = _mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_frame(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        """
        Run Holistic inference on a single BGR frame.

        Args:
            frame: OpenCV BGR image (H, W, 3).

        Returns:
            Dictionary with numpy arrays:
                - ``pose``: (33, 4) — x, y, z, visibility
                - ``left_hand``: (21, 3) — x, y, z
                - ``right_hand``: (21, 3) — x, y, z
                - ``face``: (468, 3) — x, y, z
            All zeros when a landmark group is not detected.
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected BGR frame (H, W, 3), got shape {frame.shape}")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._holistic.process(rgb)

        return {
            "pose": _landmarks_to_array(results.pose_landmarks, _NUM_POSE, _POSE_DIM),
            "left_hand": _landmarks_to_array(
                results.left_hand_landmarks, _NUM_HAND, _HAND_DIM
            ),
            "right_hand": _landmarks_to_array(
                results.right_hand_landmarks, _NUM_HAND, _HAND_DIM
            ),
            "face": _landmarks_to_array(results.face_landmarks, _NUM_FACE, _FACE_DIM),
        }

    def extract_keypoints(self, results: dict[str, np.ndarray]) -> np.ndarray:
        """
        Flatten pose and hand landmarks into a single feature vector.

        Concatenates pose (132) + left_hand (63) + right_hand (63) = 258 features.
        Face landmarks are not included in this vector.

        Args:
            results: Output from :meth:`process_frame`.

        Returns:
            float32 array of shape (258,).
        """
        pose = np.asarray(results["pose"], dtype=np.float32).reshape(-1)
        left = np.asarray(results["left_hand"], dtype=np.float32).reshape(-1)
        right = np.asarray(results["right_hand"], dtype=np.float32).reshape(-1)

        if pose.size != _POSE_FEATURES:
            raise ValueError(f"Expected pose size {_POSE_FEATURES}, got {pose.size}")
        if left.size != _HAND_FEATURES or right.size != _HAND_FEATURES:
            raise ValueError(
                f"Expected hand size {_HAND_FEATURES} each, "
                f"got left={left.size}, right={right.size}"
            )

        return np.concatenate([pose, left, right]).astype(np.float32)

    def draw_landmarks(
        self,
        frame: np.ndarray,
        results: dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Draw pose, hand, and face landmarks on a BGR frame.

        Args:
            frame: OpenCV BGR image to draw on (modified in place and returned).
            results: Landmark dict from :meth:`process_frame`.

        Returns:
            The annotated BGR frame.
        """
        output = frame.copy()
        pose = results.get("pose")
        left = results.get("left_hand")
        right = results.get("right_hand")
        face = results.get("face")

        if pose is not None and np.any(pose):
            pose_lm = _array_to_landmark_list(np.asarray(pose, dtype=np.float32))
            _mp_drawing.draw_landmarks(
                output,
                pose_lm,
                _mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=_mp_drawing_styles.get_default_pose_landmarks_style(),
            )

        if left is not None and np.any(left):
            left_lm = _array_to_landmark_list(np.asarray(left, dtype=np.float32))
            _mp_drawing.draw_landmarks(
                output,
                left_lm,
                _mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=_mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=_mp_drawing_styles.get_default_hand_connections_style(),
            )

        if right is not None and np.any(right):
            right_lm = _array_to_landmark_list(np.asarray(right, dtype=np.float32))
            _mp_drawing.draw_landmarks(
                output,
                right_lm,
                _mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=_mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=_mp_drawing_styles.get_default_hand_connections_style(),
            )

        if face is not None and np.any(face):
            face_lm = _array_to_landmark_list(np.asarray(face, dtype=np.float32))
            _mp_drawing.draw_landmarks(
                output,
                face_lm,
                _mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=_mp_drawing_styles.get_default_face_mesh_contours_style(),
            )

        return output

    def close(self) -> None:
        """Release MediaPipe Holistic resources."""
        self._holistic.close()

    def __enter__(self) -> HolisticTracker:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
