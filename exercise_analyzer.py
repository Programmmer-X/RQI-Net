"""
exercise_analyzer.py

Per-repetition quality metrics for the two exercises in scope:
  - Squat
  - Arm (Shoulder) Abduction

Consumes a list of pose_estimator.FrameLandmarks (one repetition's worth
of frames) and returns the metrics the spec's RQI formulas need:

  Squat:
    - depth_score        (0-1): how well knee angle hit the 70-100 deg target
    - posture_score       (0-1): how well hip angle hit the 60-120 deg target
    - stability_score     (0-1): inverse of frame-to-frame landmark jitter

  Arm Abduction:
    - rom_score           (0-1): how well arm elevation hit the 160-180 deg target
    - elbow_score         (0-1): how well elbow angle hit the 170-180 deg target
    - smoothness_score    (0-1): inverse of frame-to-frame angle jitter

Scores are 0-1 so rqi.py can apply the spec's weighted sums directly
(multiply by 100 there if you want the 0-100 RQI scale).
"""

from dataclasses import dataclass
from statistics import pstdev
from typing import Optional, Sequence

from angle_calculator import joint_angle, landmark_xy
from pose_estimator import FrameLandmarks


def _score_in_range(value: float, low: float, high: float) -> float:
    """
    1.0 if value is inside [low, high]. Falls off linearly outside the
    range, reaching 0.0 at a distance equal to the range's own width.
    """
    if low <= value <= high:
        return 1.0
    width = high - low
    if width <= 0:
        return 0.0
    distance = (low - value) if value < low else (value - high)
    return max(0.0, 1.0 - distance / width)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass
class SquatMetrics:
    depth_score: float
    posture_score: float
    stability_score: float
    mean_knee_angle: float
    mean_hip_angle: float


@dataclass
class ArmAbductionMetrics:
    rom_score: float
    elbow_score: float
    smoothness_score: float
    mean_elevation_angle: float
    mean_elbow_angle: float


def _best_visible_side(frames: Sequence[FrameLandmarks], joint_prefix: str) -> str:
    """
    REHAB24-6 doesn't label a side for squat (it's bilateral), so pick
    whichever side MediaPipe tracked more confidently across the rep.
    """
    left_vis = _mean([f.points[f"left_{joint_prefix}"][3] for f in frames])
    right_vis = _mean([f.points[f"right_{joint_prefix}"][3] for f in frames])
    return "left" if left_vis >= right_vis else "right"


def analyze_squat(frames: Sequence[FrameLandmarks]) -> SquatMetrics:
    if not frames:
        raise ValueError("No frames provided for squat analysis.")

    side = _best_visible_side(frames, "knee")

    knee_angles = []
    hip_angles = []
    for f in frames:
        p = f.points
        # Knee angle: hip-knee-ankle.
        knee_angles.append(
            joint_angle(
                landmark_xy_from(p, f"{side}_hip"),
                landmark_xy_from(p, f"{side}_knee"),
                landmark_xy_from(p, f"{side}_ankle"),
            )
        )
        # Hip angle: shoulder-hip-knee.
        hip_angles.append(
            joint_angle(
                landmark_xy_from(p, f"{side}_shoulder"),
                landmark_xy_from(p, f"{side}_hip"),
                landmark_xy_from(p, f"{side}_knee"),
            )
        )

    mean_knee = _mean(knee_angles)
    mean_hip = _mean(hip_angles)

    depth_score = _score_in_range(mean_knee, 70, 100)
    posture_score = _score_in_range(mean_hip, 60, 120)

    # Stability: penalize high frame-to-frame variance in the knee angle.
    jitter = pstdev(knee_angles) if len(knee_angles) > 1 else 0.0
    stability_score = max(0.0, 1.0 - jitter / 15.0)  # 15 deg stdev -> score 0

    return SquatMetrics(
        depth_score=depth_score,
        posture_score=posture_score,
        stability_score=stability_score,
        mean_knee_angle=mean_knee,
        mean_hip_angle=mean_hip,
    )


def analyze_arm_abduction(
    frames: Sequence[FrameLandmarks], side: Optional[str] = None
) -> ArmAbductionMetrics:
    """
    side: "left" or "right". REHAB24-6 reps are single-arm — pass the
    side from Segmentation.csv's `exercise_subtype` column
    ("left arm" / "right arm"). If not given, falls back to whichever
    side MediaPipe tracked more confidently (useful for non-REHAB24-6
    footage where the side isn't known in advance).
    """
    if not frames:
        raise ValueError("No frames provided for arm abduction analysis.")

    if side is None:
        side = _best_visible_side(frames, "wrist")
    elif side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    elevation_angles = []
    elbow_angles = []
    for f in frames:
        p = f.points
        # Elevation angle: hip-shoulder-wrist (how far the arm has raised
        # from the torso line).
        elevation_angles.append(
            joint_angle(
                landmark_xy_from(p, f"{side}_hip"),
                landmark_xy_from(p, f"{side}_shoulder"),
                landmark_xy_from(p, f"{side}_wrist"),
            )
        )
        # Elbow angle: shoulder-elbow-wrist (straightness).
        elbow_angles.append(
            joint_angle(
                landmark_xy_from(p, f"{side}_shoulder"),
                landmark_xy_from(p, f"{side}_elbow"),
                landmark_xy_from(p, f"{side}_wrist"),
            )
        )

    mean_elevation = _mean(elevation_angles)
    mean_elbow = _mean(elbow_angles)

    rom_score = _score_in_range(mean_elevation, 160, 180)
    elbow_score = _score_in_range(mean_elbow, 170, 180)

    jitter = pstdev(elevation_angles) if len(elevation_angles) > 1 else 0.0
    smoothness_score = max(0.0, 1.0 - jitter / 15.0)

    return ArmAbductionMetrics(
        rom_score=rom_score,
        elbow_score=elbow_score,
        smoothness_score=smoothness_score,
        mean_elevation_angle=mean_elevation,
        mean_elbow_angle=mean_elbow,
    )


def landmark_xy_from(points: dict, name: str):
    """points[name] is (x, y, z, visibility) — take just (x, y)."""
    x, y, _z, _vis = points[name]
    return (x, y)


if __name__ == "__main__":
    # Smoke test with synthetic frames (no real video needed).
    def fake_frame(i, knee_bend_deg):
        # Build a synthetic stick figure: hip at origin, knee below,
        # ankle further below, bent by `knee_bend_deg` from straight.
        import math

        hip = (0.5, 0.5)
        knee = (0.5, 0.65)
        angle_rad = math.radians(180 - knee_bend_deg)
        ankle = (0.5 + 0.15 * math.sin(angle_rad), 0.65 + 0.15 * math.cos(angle_rad))
        shoulder = (0.5, 0.3)
        elbow = (0.6, 0.3)
        wrist = (0.7, 0.3)
        pts = {
            "left_hip": (*hip, 0.0, 1.0),
            "left_knee": (*knee, 0.0, 1.0),
            "left_ankle": (*ankle, 0.0, 1.0),
            "left_shoulder": (*shoulder, 0.0, 1.0),
            "left_elbow": (*elbow, 0.0, 1.0),
            "left_wrist": (*wrist, 0.0, 1.0),
            "right_hip": (*hip, 0.0, 1.0),
            "right_knee": (*knee, 0.0, 1.0),
            "right_ankle": (*ankle, 0.0, 1.0),
            "right_shoulder": (*shoulder, 0.0, 1.0),
            "right_elbow": (*elbow, 0.0, 1.0),
            "right_wrist": (*wrist, 0.0, 1.0),
        }
        return FrameLandmarks(frame_index=i, timestamp_sec=i / 30.0, points=pts)

    synthetic_frames = [fake_frame(i, knee_bend_deg=85) for i in range(10)]
    squat_result = analyze_squat(synthetic_frames)
    print("Squat metrics:", squat_result)
