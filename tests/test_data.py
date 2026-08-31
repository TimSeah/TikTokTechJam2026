from pathlib import Path

from PIL import Image

from src.detector.data import build_manifests


def _write_images(root: Path, split: str, class_name: str, count: int) -> None:
    class_dir = root / split / class_name
    class_dir.mkdir(parents=True)
    for index in range(count):
        Image.new("RGB", (32, 32), color=(index, index, index)).save(
            class_dir / f"{index}.jpg"
        )


def test_build_manifests_is_balanced_and_reproducible(tmp_path: Path) -> None:
    data_root = tmp_path / "cifake"
    for split, count in (("train", 5), ("test", 2)):
        for class_name in ("REAL", "FAKE"):
            _write_images(data_root, split, class_name, count)

    first = build_manifests(data_root, tmp_path / "first", seed=7, dev_per_class=2)
    second = build_manifests(data_root, tmp_path / "second", seed=7, dev_per_class=2)

    assert len(first["train"]) == 10
    assert len(first["test"]) == 4
    assert len(first["development"]) == 4
    assert [record.image_id for record in first["development"]] == [
        record.image_id for record in second["development"]
    ]
    assert {record.label for record in first["development"]} == {0, 1}
    assert {record.role for record in first["development"]} == {"development"}
