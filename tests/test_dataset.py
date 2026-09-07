from unittest.mock import patch

import pytest

from src.dataset import atomic_write, read_labels, read_names, write_labels


@pytest.mark.parametrize(
    "text",
    [
        'names: [player, "ball: white"]',
        'names:\n  1: "ball: white"\n  0: player # comment',
    ],
)
def test_yaml_names(tmp_path, text):
    path = tmp_path / "data.yaml"
    path.write_text(text)
    assert read_names(path) == {0: "player", 1: "ball: white"}


@pytest.mark.parametrize(
    "text", ["names: []", "names: {1: ball}", "names: null", "names: {0: 3}", 'names: {"0": ball}']
)
def test_bad_names(tmp_path, text):
    path = tmp_path / "data.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_names(path)


def test_label_roundtrip_and_clipping(tmp_path):
    path = tmp_path / "labels" / "frame.txt"
    shape = (101, 203, 3)
    boxes = [(1, (12.25, 5.5, 130.75, 90.125))]
    write_labels(path, boxes, shape)
    read = read_labels(path, shape, {0: "player", 1: "ball"})
    assert read[0][1] == pytest.approx(boxes[0][1], abs=0.001)
    write_labels(path, [(0, (-5, -5, 300, 200))], shape)
    assert path.read_text() == "0 0.500000 0.500000 1.000000 1.000000\n"
    write_labels(path, [], shape)
    assert read_labels(path, shape, {0: "player"}) == []


@pytest.mark.parametrize(
    "line",
    [
        "0 1 2",
        "0 nan .5 .1 .1",
        "-1 .5 .5 .1 .1",
        "0.5 .5 .5 .1 .1",
        "2 .5 .5 .1 .1",
        "0 .5 .5 0 .1",
        "0 .5 .5 -1 .1",
        "0 inf .5 .1 .1",
        "bad label",
    ],
)
def test_invalid_labels_are_not_silently_dropped(tmp_path, line):
    path = tmp_path / "frame.txt"
    path.write_text(line)
    with pytest.raises(ValueError, match="frame.txt:1"):
        read_labels(path, (100, 100), {0: "player"})
    assert path.read_text() == line


def test_atomic_save_failure_preserves_labels(tmp_path):
    path = tmp_path / "frame.txt"
    path.write_text("reviewed labels")
    with patch("src.dataset.os.replace", side_effect=OSError("disk error")):
        with pytest.raises(OSError):
            atomic_write(path, "new labels")
    assert path.read_text() == "reviewed labels"
    assert list(tmp_path.iterdir()) == [path]
