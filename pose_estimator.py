"""
pose_estimator.py

Wraps MediaPipe Pose (legacy `solutions` API, mediapipe==0.10.21 pinned —
see README.md for why the version is pinned) to extract per-frame landmarks
from a video file.

NOTE ON MEDIAPIPE VERSIONING:
mediapipe>=1.0.0 removed the `mp.solutions.pose` API entirely in favor of
the newer Tasks API (`mediapipe.tasks.python.vision.PoseLandmarker`), which
requires downloading a separate .task model bundle. This project pins
mediapipe==0.10.21 to keep the simpler solutions API. If you upgrade
mediapipe later, this file needs a rewrite against the Tasks API.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

# The 12 landmarks the spec's joint-angle/exercise modules need.
# (Full MediaPipe Pose gives 33; we keep the indices for the ones we use.)
LANDMARKS_OF_INTEREST = {
    "left_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "left_elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
    "right_elbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
    "left_wrist": mp_pose.PoseLandmark.LEFT_WRIST,
    "right_wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
    "left_hip": mp_pose.PoseLandmark.LEFT_HIP,
    "right_hip": mp_pose.PoseLandmark.RIGHT_HIP,
    "left_knee": mp_pose.PoseLandmark.LEFT_KNEE,
    "right_knee": mp_pose.PoseLandmark.RIGHT_KNEE,
    "left_ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
    "right_ankle": mp_pose.PoseLandmark.RIGHT_ANKLE,
}


@dataclass
class FrameLandmarks:
    frame_index: int
    timestamp_sec: float
    points: dict  # name -> (x, y, z, visibility), normalized [0,1] image coords


def extract_video_landmarks(
    video_path: str,
    model_complexity: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    max_frames: Optional[int] = None,
) -> list[FrameLandmarks]:
    """
    Run MediaPipe Pose over every frame of a video and return the
    landmarks of interest per frame. Frames with no detected person
    are skipped (not padded) — callers should handle gaps.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    results_out: list[FrameLandmarks] = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as pose:
        frame_index = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_index >= max_frames:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            result = pose.process(frame_rgb)

            if result.pose_landmarks is not None:
                lm = result.pose_landmarks.landmark
                points = {
                    name: (lm[idx].x, lm[idx].y, lm[idx].z, lm[idx].visibility)
                    for name, idx in LANDMARKS_OF_INTEREST.items()
                }
                results_out.append(
                    FrameLandmarks(
                        frame_index=frame_index,
                        timestamp_sec=frame_index / fps,
                        points=points,
                    )
                )
            frame_index += 1

    cap.release()
    return results_out


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python pose_estimator.py <path_to_video>")
        sys.exit(1)

    frames = extract_video_landmarks(sys.argv[1], max_frames=30)
    print(f"Extracted landmarks for {len(frames)} frames (capped at 30 for this smoke test).")
    if frames:
        print("Example frame 0 points:", frames[0].points)
