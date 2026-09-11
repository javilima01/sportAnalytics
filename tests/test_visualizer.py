from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from src.dataset import read_labels
from src.visualizer import Visualizer


@pytest.fixture
def dataset(tmp_path):
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "data.yaml").write_text("names: [player, ball]\n")
    for name in ("a.JPG", "b.jpeg"):
        assert cv2.imwrite(
            str(tmp_path / "images" / "train" / name), np.zeros((80, 100, 3), np.uint8)
        )
    return tmp_path


@pytest.fixture
def gui(monkeypatch):
    # Exercise actual editor event handling without requiring a display server.
    for method in ("namedWindow", "setMouseCallback", "imshow", "destroyAllWindows"):
        monkeypatch.setattr(cv2, method, Mock())
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *args: 1)


def draw_box(viewer):
    viewer._mouse_callback(cv2.EVENT_LBUTTONDOWN, 10, 10, 0, None)
    viewer._mouse_callback(cv2.EVENT_MOUSEMOVE, 50, 50, 0, None)
    viewer._mouse_callback(cv2.EVENT_LBUTTONUP, 50, 50, 0, None)


def test_preview_includes_unlabeled_images(dataset, tmp_path):
    viewer = Visualizer(dataset, max_images=20)
    output = tmp_path / "preview"
    viewer.visualize(output)
    assert len(list(output.iterdir())) == 2
    with pytest.raises(ValueError, match="outside"):
        viewer.visualize(viewer.image_dir)


def test_edit_autosaves_and_changes_class(dataset, gui, monkeypatch):
    viewer = Visualizer(dataset)

    def first_key(delay):
        draw_box(viewer)
        return ord("c")

    keys = iter([first_key, lambda _: ord("n"), lambda _: 27])
    monkeypatch.setattr(cv2, "waitKeyEx", lambda delay: next(keys)(delay))
    viewer.edit()
    labels = read_labels(dataset / "labels/train/a.txt", (80, 100), viewer.names)
    assert labels[0][0] == 1
    assert labels[0][1] == pytest.approx((10, 10, 50, 50))
    assert viewer.index == 1
    cv2.destroyAllWindows.assert_called_once()


@pytest.mark.parametrize("close_window", [False, True])
def test_exit_and_window_close_save(dataset, gui, monkeypatch, close_window):
    viewer = Visualizer(dataset)
    calls = 0

    def key(delay):
        nonlocal calls
        calls += 1
        draw_box(viewer)
        return -1 if close_window else 27

    monkeypatch.setattr(cv2, "waitKeyEx", key)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *args: 0 if close_window and calls else 1)
    viewer.edit()
    assert (dataset / "labels/train/a.txt").exists()


def test_navigation_boundaries_and_windows_arrow(dataset, gui, monkeypatch):
    viewer = Visualizer(dataset)
    keys = iter([ord("p"), 2555904, ord("n"), 2424832, 27])
    monkeypatch.setattr(cv2, "waitKeyEx", lambda delay: next(keys))
    viewer.edit()
    assert viewer.index == 0


def test_mouse_delete_resets_selection(dataset):
    viewer = Visualizer(dataset)
    viewer.current_image = np.zeros((80, 100, 3), np.uint8)
    draw_box(viewer)
    viewer._mouse_callback(cv2.EVENT_MBUTTONDOWN, 20, 20, 0, None)
    assert viewer.selected_box is None
    assert viewer.boxes == []
    viewer._mouse_callback(cv2.EVENT_LBUTTONDOWN, 70, 70, cv2.EVENT_FLAG_CTRLKEY, None)
    assert viewer.start_point is None


def test_delete_last_image(dataset, gui, monkeypatch):
    viewer = Visualizer(dataset)
    monkeypatch.setattr(cv2, "waitKeyEx", lambda delay: ord("d"))
    viewer.edit()
    assert viewer.image_paths == []
    assert not list(viewer.image_dir.iterdir())


def test_malformed_labels_untouched_on_editor_error(dataset, gui):
    label = dataset / "labels/train/a.txt"
    label.parent.mkdir(parents=True)
    label.write_text("bad labels")
    viewer = Visualizer(dataset)
    with pytest.raises(ValueError, match="invalid YOLO"):
        viewer.edit()
    assert label.read_text() == "bad labels"
    cv2.destroyAllWindows.assert_called_once()


def test_roboflow_images_without_labels(tmp_path):
    images = tmp_path / "valid/images"
    images.mkdir(parents=True)
    (tmp_path / "data.yaml").write_text("names: {0: ball}")
    viewer = Visualizer(tmp_path, split="valid")
    assert viewer.image_dir == images
    assert viewer.label_dir == tmp_path / "valid/labels"


def test_prediction_preview_draws_model_boxes(dataset, tmp_path, monkeypatch):
    boxes = Mock()
    boxes.cls.tolist.return_value = [1]
    boxes.conf.tolist.return_value = [0.9]
    boxes.xyxy.tolist.return_value = [[10, 10, 50, 50]]
    model = Mock(task="detect", names={0: "player", 1: "ball"})
    model.predict.return_value = [Mock(boxes=boxes)]
    monkeypatch.setattr("src.visualizer.YOLO", Mock(return_value=model))
    viewer = Visualizer(dataset, max_images=20, model_path="model.pt", conf=0.4, imgsz=640)
    output = tmp_path / "preview"
    viewer.visualize(output)
    assert len(list(output.iterdir())) == 2
    assert model.predict.call_args.kwargs["conf"] == 0.4
    assert model.predict.call_args.kwargs["imgsz"] == 640
    assert model.predict.call_args.kwargs["verbose"] is False


def test_prediction_preview_rejects_bad_options(dataset):
    with pytest.raises(ValueError, match=r"\.pt"):
        Visualizer(dataset, model_path="model.onnx")
    with pytest.raises(ValueError, match="conf"):
        Visualizer(dataset, model_path="model.pt", conf=2)
    with pytest.raises(ValueError, match="imgsz"):
        Visualizer(dataset, model_path="model.pt", imgsz=8)


def test_prediction_preview_is_read_only(dataset, monkeypatch):
    model = Mock(task="detect", names={0: "player"})
    monkeypatch.setattr("src.visualizer.YOLO", Mock(return_value=model))
    viewer = Visualizer(dataset, model_path="model.pt")
    with pytest.raises(ValueError, match="read-only"):
        viewer.edit()
