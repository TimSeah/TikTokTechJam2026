from pathlib import Path

from PIL import Image

from src.detector.acquire_native_data import (
    assign_split_roles,
    download_selected_row,
    fetch_pages,
    read_excluded_hashes,
    read_excluded_ids,
    safe_name,
    select_balanced_rows,
)
from src.detector.data import ImageRecord


def _page(offset: int, labels: list[int]) -> dict:
    return {
        "rows": [
            {"row_idx": offset + index, "row": {"label": label}}
            for index, label in enumerate(labels)
        ]
    }


def test_select_balanced_rows_is_deterministic_and_ignores_tampered() -> None:
    pages = [
        _page(0, [0, 1, 2, 0, 1]),
        _page(100, [2, 1, 0, 1, 0]),
    ]

    first = select_balanced_rows(pages, per_class=2, seed=7)
    second = select_balanced_rows(pages, per_class=2, seed=7)

    assert first == second
    assert [row["row"]["label"] for row in first] == [0, 0, 1, 1]


def test_select_balanced_rows_stops_after_filling_quotas() -> None:
    def pages():
        yield _page(0, [0, 1])
        raise AssertionError("selection fetched an unnecessary page")

    selected = select_balanced_rows(pages(), per_class=1, seed=2026)

    assert len(selected) == 2


def test_fetch_pages_caches_metadata_for_resume(tmp_path: Path, monkeypatch) -> None:
    first_page = {"rows": [{"row_idx": 0}]}

    def fetch_page_once(session, dataset, config, split, offset):
        assert offset == 100
        return {"rows": [{"row_idx": 100}]}

    monkeypatch.setattr("src.detector.acquire_native_data.fetch_page", fetch_page_once)
    pages = list(fetch_pages([0, 100], first_page, workers=1, cache_dir=tmp_path))

    assert [page["rows"][0]["row_idx"] for page in pages] == [0, 100]

    def fail_fetch(*args):
        raise AssertionError("cached metadata should not be fetched again")

    monkeypatch.setattr("src.detector.acquire_native_data.fetch_page", fail_fetch)
    resumed = list(fetch_pages([0, 100], first_page, workers=1, cache_dir=tmp_path))

    assert resumed == pages


def test_select_balanced_rows_excludes_validation_ids() -> None:
    pages = [
        {
            "rows": [
                {"row_idx": 0, "row": {"label": 0, "img_id": "excluded"}},
                {"row_idx": 1, "row": {"label": 0, "img_id": "real"}},
                {"row_idx": 2, "row": {"label": 1, "img_id": "fake"}},
            ]
        }
    ]

    selected = select_balanced_rows(
        pages, per_class=1, seed=2026, excluded_ids={"excluded"}
    )

    assert {row["row"]["img_id"] for row in selected} == {"real", "fake"}


def test_read_excluded_ids_accepts_blind_and_training_manifests(
    tmp_path: Path,
) -> None:
    blind = tmp_path / "blind.csv"
    blind.write_text("item_id,label\nblind-id,0\n", encoding="utf-8")
    training = tmp_path / "training.csv"
    training.write_text("img_id,label\ntraining-id,1\n", encoding="utf-8")

    assert read_excluded_ids([blind, training]) == {"blind-id", "training-id"}


def test_read_excluded_hashes_ignores_blank_values(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "item_id,sha256\none,ABCDEF\ntwo,\n", encoding="utf-8"
    )

    assert read_excluded_hashes([manifest_path]) == {"abcdef"}


def test_safe_name_removes_path_characters() -> None:
    assert safe_name("folder/image id:1") == "folder_image_id_1"


def test_download_selected_row_reuses_existing_image_without_network(
    tmp_path: Path, monkeypatch,
) -> None:
    destination = tmp_path / "REAL" / "000007_existing.jpg"
    destination.parent.mkdir(parents=True)
    Image.new("RGB", (8, 6), "red").save(destination)

    def fail_session():
        raise AssertionError("network should not be used for an existing image")

    monkeypatch.setattr(
        "src.detector.acquire_native_data.worker_session", fail_session
    )
    record, provenance = download_selected_row(
        {
            "row_idx": 7,
            "row": {
                "label": 0,
                "img_id": "existing",
                "image": {"src": "https://example.invalid/image.jpg"},
            },
        },
        tmp_path,
        revision="revision",
        seed=2026,
    )

    assert record.relative_path == "REAL/000007_existing.jpg"
    assert provenance["reused_existing"] is True


def test_assign_split_roles_uses_reserves_after_failed_candidates() -> None:
    downloads = []
    for label, class_name in ((0, "REAL"), (1, "FAKE")):
        for index in range(4):
            record = ImageRecord(
                image_id=f"{class_name}-{index}",
                relative_path=f"{class_name}/{index}.jpg",
                source_split="train",
                role="train",
                label=label,
                class_name=class_name,
            )
            downloads.append(
                (
                    record,
                    {"label": label, "row_idx": index, "sha256": f"{label}-{index}"},
                )
            )

    records, train, evaluation, provenance = assign_split_roles(downloads, 2, 1)

    assert len(records) == 6
    assert len(train) == 4
    assert len(evaluation) == 2
    assert [row["role"] for row in provenance].count("train") == 4


def test_assign_split_roles_excludes_frozen_and_duplicate_hashes() -> None:
    downloads = []
    for label, class_name in ((0, "REAL"), (1, "FAKE")):
        for index, digest in enumerate(("frozen", "duplicate", "duplicate", "unique")):
            record = ImageRecord(
                image_id=f"{class_name}-{index}",
                relative_path=f"{class_name}/{index}.jpg",
                source_split="train",
                role="train",
                label=label,
                class_name=class_name,
            )
            downloads.append((record, {"label": label, "sha256": f"{label}-{digest}"}))

    records, _, _, provenance = assign_split_roles(
        downloads, train_per_class=2, eval_per_class=0, excluded_hashes={"0-frozen"}
    )

    assert len(records) == 4
    assert len({row["sha256"] for row in provenance}) == 4
    assert "0-frozen" not in {row["sha256"] for row in provenance}
