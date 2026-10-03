"""
angle_calculator.py

Computes the joint angle at point B given three 2D/3D points A-B-C:

    theta = arccos( (BA . BC) / (|BA| |BC|) )

Works with MediaPipe landmarks (x, y, [z]) or plain (x, y) tuples.
"""

import numpy as np


def joint_angle(a, b, c) -> float:
    """
    Angle at vertex b (in degrees), formed by rays b->a and b->c.

    a, b, c: array-like of length 2 (x, y) or 3 (x, y, z).
    Returns the angle in degrees, in [0, 180].
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        raise ValueError("Degenerate points: zero-length vector at vertex b.")

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)  # guard against float rounding past +/-1
    return float(np.degrees(np.arccos(cosine)))


def landmark_xy(landmark) -> tuple:
    """Extract (x, y) from a MediaPipe NormalizedLandmark-like object."""
    return (landmark.x, landmark.y)


def landmark_xyz(landmark) -> tuple:
    """Extract (x, y, z) from a MediaPipe NormalizedLandmark-like object."""
    return (landmark.x, landmark.y, landmark.z)


if __name__ == "__main__":
    # Sanity check: a right angle.
    a = (0, 1)
    b = (0, 0)
    c = (1, 0)
    print(f"Expected 90.0, got {joint_angle(a, b, c):.2f}")

    # Sanity check: a straight line (180 degrees).
    a = (-1, 0)
    b = (0, 0)
    c = (1, 0)
    print(f"Expected 180.0, got {joint_angle(a, b, c):.2f}")
