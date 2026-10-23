"""Ball position in the camera frame from a bounding box.

Camera frame follows OpenCV: x right, y down, z forward (optical axis).
Two monocular range estimates are combined:

* size:   the ball's angular diameter vs. its known physical diameter.
          Needs only intrinsics; error grows ~ range^2 because the ball shrinks to a few pixels.
* ground: the ball rests on a flat floor, so the ray through its centre is intersected
          with a plane (camera height - ball radius) below the camera. Needs mount height and
          pitch; error grows ~ range^2 / height with pitch error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

# ITF regulation tennis ball diameter is 6.54-6.86 cm.
BALL_DIAMETER_M = 0.067


@dataclass
class CameraModel:
    K: np.ndarray  # 3x3 intrinsics
    dist: np.ndarray  # OpenCV distortion coefficients
    width: int
    height: int
    calibrated: bool = True

    @classmethod
    def load(cls, path: str | Path) -> "CameraModel":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(
            K=np.array(data["camera_matrix"], dtype=np.float64).reshape(3, 3),
            dist=np.array(data["dist_coeffs"], dtype=np.float64).reshape(-1),
            width=int(data["image_width"]),
            height=int(data["image_height"]),
        )

    def save(self, path: str | Path, extra: Optional[dict] = None) -> None:
        data = {
            "image_width": self.width,
            "image_height": self.height,
            "camera_matrix": self.K.tolist(),
            "dist_coeffs": self.dist.reshape(-1).tolist(),
            **(extra or {}),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float) -> "CameraModel":
        """Rough pinhole model for an uncalibrated camera. Distances will be off."""
        fx = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
        K = np.array([[fx, 0, width / 2], [0, fx, height / 2], [0, 0, 1]], dtype=np.float64)
        return cls(K=K, dist=np.zeros(5), width=width, height=height, calibrated=False)

    def scaled_to(self, width: int, height: int) -> "CameraModel":
        """Intrinsics for the same sensor at another resolution (same aspect ratio)."""
        if (width, height) == (self.width, self.height):
            return self
        sx, sy = width / self.width, height / self.height
        K = self.K.copy()
        K[0] *= sx
        K[1] *= sy
        return CameraModel(K, self.dist.copy(), width, height, self.calibrated)

    def rays(self, pixels: np.ndarray) -> np.ndarray:
        """Undistorted unit rays (N, 3) in the camera frame for pixel coords (N, 2)."""
        pts = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
        norm = cv2.undistortPoints(pts, self.K, self.dist).reshape(-1, 2)
        rays = np.column_stack([norm, np.ones(len(norm))])
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)


@dataclass(frozen=True)
class Mount:
    """Camera mount relative to a flat floor."""

    height_m: float  # optical centre above the floor
    pitch_deg: float  # downward tilt of the optical axis, positive = looking down


@dataclass(frozen=True)
class BallPosition:
    xyz: np.ndarray  # metres, camera frame
    azimuth_deg: float  # positive = ball to the right
    elevation_deg: float  # positive = ball above the optical axis
    range_size: Optional[float]
    range_ground: Optional[float]

    @property
    def range(self) -> float:
        return float(np.linalg.norm(self.xyz))


def _rot_level_from_camera(pitch_rad: float) -> np.ndarray:
    """Rotation taking camera-frame vectors into a gravity-level frame (x right, y down, z forward)."""
    c, s = math.cos(pitch_rad), math.sin(pitch_rad)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def floor_xy(xyz_cam: np.ndarray, mount: Optional[Mount]) -> tuple[float, float]:
    """(right, forward) in metres on the floor plane, for top-down plots."""
    level = _rot_level_from_camera(math.radians(mount.pitch_deg)) @ xyz_cam if mount else xyz_cam
    return float(level[0]), float(level[2])


def range_from_size(camera: CameraModel, x1: float, x2: float, y: float, diameter_m: float) -> Optional[float]:
    """Range to the ball centre from the angle subtended by its horizontal extent."""
    left, right = camera.rays(np.array([[x1, y], [x2, y]]))
    alpha = math.acos(float(np.clip(left @ right, -1.0, 1.0)))
    if alpha <= 1e-6:
        return None
    return (diameter_m / 2) / math.sin(alpha / 2)


def range_from_ground(camera: CameraModel, u: float, v: float, mount: Mount, radius_m: float) -> Optional[float]:
    """Range to the ball centre by intersecting its ray with the plane of ball centres."""
    ray = camera.rays(np.array([[u, v]]))[0]
    ray_level = _rot_level_from_camera(math.radians(mount.pitch_deg)) @ ray
    drop = mount.height_m - radius_m
    if ray_level[1] <= 1e-6 or drop <= 0:  # ray at or above the horizon
        return None
    return drop / ray_level[1]


def _fuse(range_size: Optional[float], range_ground: Optional[float], camera: CameraModel, mount: Optional[Mount]) -> Optional[float]:
    if range_size is None or range_ground is None or mount is None:
        return range_size if range_size is not None else range_ground
    # Inverse-variance weighting with first-order error models:
    #   size:   dr = r^2 / (f * D) * px_err
    #   ground: dr = r^2 / h * pitch_err
    r = (range_size + range_ground) / 2
    f = camera.K[0, 0]
    sigma_size = r * r / (f * BALL_DIAMETER_M) * 1.5  # ~1.5 px box-edge error
    sigma_ground = r * r / mount.height_m * math.radians(1.0)  # ~1 deg pitch error
    w_size, w_ground = 1 / sigma_size**2, 1 / sigma_ground**2
    return (w_size * range_size + w_ground * range_ground) / (w_size + w_ground)


def locate_ball(
    camera: CameraModel,
    box_xyxy: tuple[float, float, float, float],
    mount: Optional[Mount] = None,
    diameter_m: float = BALL_DIAMETER_M,
) -> Optional[BallPosition]:
    x1, y1, x2, y2 = box_xyxy
    u, v = (x1 + x2) / 2, (y1 + y2) / 2
    ray = camera.rays(np.array([[u, v]]))[0]

    r_size = range_from_size(camera, x1, x2, v, diameter_m)
    r_ground = range_from_ground(camera, u, v, mount, diameter_m / 2) if mount else None
    r = _fuse(r_size, r_ground, camera, mount)
    if r is None:
        return None

    return BallPosition(
        xyz=ray * r,
        azimuth_deg=math.degrees(math.atan2(ray[0], ray[2])),
        elevation_deg=math.degrees(math.atan2(-ray[1], math.hypot(ray[0], ray[2]))),
        range_size=r_size,
        range_ground=r_ground,
    )
