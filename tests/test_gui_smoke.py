"""Opt-in native desktop smoke test; the normal suite simulates GUI input."""

import os

import cv2
import numpy as np
import pytest

from src.dataset import read_labels
from src.visualizer import Visualizer


@pytest.mark.gui
@pytest.mark.skipif(
    os.environ.get("RUN_GUI_SMOKE") != "1", reason="Requires an interactive desktop"
)
def test_native_editor_window(tmp_path, monkeypatch):
    images = tmp_path / "images/train"
    images.mkdir(parents=True)
    (tmp_path / "data.yaml").write_text("names: [player, ball]")
    cv2.imwrite(str(images / "frame.jpg"), np.full((240, 320, 3), 80, np.uint8))
    viewer = Visualizer(tmp_path, window_name="Image tagging smoke test")
    real_wait = cv2.waitKeyEx
    calls = 0

    def interact(delay):
        nonlocal calls
        real_wait(300)
        calls += 1
        if calls == 1:
            viewer._mouse_callback(cv2.EVENT_LBUTTONDOWN, 40, 40, 0, None)
            viewer._mouse_callback(cv2.EVENT_MOUSEMOVE, 200, 180, 0, None)
            viewer._mouse_callback(cv2.EVENT_LBUTTONUP, 200, 180, 0, None)
            return ord("c")
        return 27

    monkeypatch.setattr(cv2, "waitKeyEx", interact)
    viewer.edit()
    boxes = read_labels(tmp_path / "labels/train/frame.txt", (240, 320), viewer.names)
    assert len(boxes) == 1 and boxes[0][0] == 1
    assert boxes[0][1] == pytest.approx((40, 40, 200, 180), abs=0.001)
