import cv2
import numpy as np
import pytest

from helloballs.vision.calibration import CalibrationSession, find_corners

PATTERN = (9, 6)
SQUARE = 0.025
K_TRUE = np.array([[800.0, 0, 640], [0, 800.0, 360], [0, 0, 1]])


def render_view(rvec, tvec, size=(1280, 720)):
    """Render a checkerboard seen from pose (rvec, tvec) by an ideal pinhole camera."""
    cols, rows = PATTERN[0] + 1, PATTERN[1] + 1
    px = 40
    board = np.full(((rows + 2) * px, (cols + 2) * px), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                board[(r + 1) * px : (r + 2) * px, (c + 1) * px : (c + 2) * px] = 0
    # board pixel -> board metres (origin at the first inner corner)
    m = SQUARE / px
    corners_board_px = np.float32([[0, 0], [board.shape[1], 0], [board.shape[1], board.shape[0]], [0, board.shape[0]]])
    corners_m = np.column_stack([(corners_board_px - 2 * px) * m, np.zeros(4)]).astype(np.float32)
    proj, _ = cv2.projectPoints(corners_m, rvec, tvec, K_TRUE, None)
    H = cv2.getPerspectiveTransform(corners_board_px, proj.reshape(4, 2).astype(np.float32))
    return cv2.warpPerspective(board, H, size, borderValue=200)


def test_calibration_recovers_intrinsics():
    rng = np.random.default_rng(0)
    session = CalibrationSession(PATTERN, SQUARE)
    for _ in range(15):
        rvec = rng.uniform(-0.4, 0.4, 3)
        tvec = np.array([rng.uniform(-0.15, 0.05), rng.uniform(-0.1, 0.0), rng.uniform(0.35, 0.6)])
        img = render_view(rvec, tvec)
        corners = find_corners(img, PATTERN)
        if corners is not None:
            session.add(corners, (img.shape[1], img.shape[0]))
    assert len(session.views) >= 10
    result = session.run()
    assert result.rms < 0.5
    assert result.camera.K[0, 0] == pytest.approx(800, rel=0.02)
    assert result.camera.K[0, 2] == pytest.approx(640, abs=10)


def test_too_few_views():
    with pytest.raises(ValueError):
        CalibrationSession().run()
