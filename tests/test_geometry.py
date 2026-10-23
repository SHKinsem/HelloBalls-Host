import math

import numpy as np
import pytest

from helloballs.vision.geometry import BALL_DIAMETER_M, CameraModel, Mount, locate_ball

K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
CAMERA = CameraModel(K=K, dist=np.zeros(5), width=1280, height=720)
R = BALL_DIAMETER_M / 2


def ball_box(center_cam: np.ndarray) -> tuple[float, float, float, float]:
    """Approximate pixel box of a ball centred at `center_cam` (exact horizontally when y == 0)."""
    x, y, z = center_cam
    ax = math.atan2(x, z)
    half = math.asin(R / math.hypot(x, z))  # angular radius seen in the xz-plane
    u_left = K[0, 2] + K[0, 0] * math.tan(ax - half)
    u_right = K[0, 2] + K[0, 0] * math.tan(ax + half)
    v = K[1, 2] + K[1, 1] * y / z
    du = (u_right - u_left) / 2
    return u_left, v - du, u_right, v + du


@pytest.mark.parametrize("dist", [0.5, 1.0, 2.0, 4.0])
def test_size_range_on_axis(dist):
    pos = locate_ball(CAMERA, ball_box(np.array([0.0, 0.0, dist])))
    assert pos.range == pytest.approx(dist, rel=0.01)
    assert pos.azimuth_deg == pytest.approx(0.0, abs=0.01)


def test_azimuth_sign_and_value():
    center = np.array([1.0, 0.0, 2.0])
    pos = locate_ball(CAMERA, ball_box(center))
    assert pos.azimuth_deg == pytest.approx(math.degrees(math.atan2(1, 2)), abs=0.05)
    assert pos.range == pytest.approx(np.linalg.norm(center), rel=0.02)


def test_ground_range_matches_geometry():
    mount = Mount(height_m=0.30, pitch_deg=20.0)
    # Ball on the floor 1.5 m ahead (level frame), expressed in the pitched camera frame.
    p = math.radians(mount.pitch_deg)
    level = np.array([0.2, mount.height_m - R, 1.5])
    cam = np.array([level[0], math.cos(p) * level[1] - math.sin(p) * level[2],
                    math.sin(p) * level[1] + math.cos(p) * level[2]])
    pos = locate_ball(CAMERA, ball_box(cam), mount)
    assert pos.range_ground == pytest.approx(np.linalg.norm(cam), rel=1e-3)
    assert pos.range == pytest.approx(np.linalg.norm(cam), rel=0.02)


def test_ground_ignored_above_horizon():
    mount = Mount(height_m=0.30, pitch_deg=0.0)
    pos = locate_ball(CAMERA, ball_box(np.array([0.0, -0.2, 2.0])), mount)
    assert pos.range_ground is None
    assert pos.range == pytest.approx(pos.range_size)


def test_scaled_intrinsics():
    half = CAMERA.scaled_to(640, 360)
    assert half.K[0, 0] == pytest.approx(450)
    assert half.K[1, 2] == pytest.approx(180)
