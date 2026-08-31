from pathlib import Path

from src.detector.acquire_wildfake_data import (
    assign_training_records,
    select_group_balanced_rows,
)
from src.detector.data import ImageRecord


def _source_rows() -> list[dict[str, str]]:
    rows = []
    for split, groups in (
        ("real", ("real-a", "real-b")),
        ("fake", ("fake-a", "fake-b")),
    ):
        for group in groups:
            for index in range(4):
                rows.append(
                    {
                        "split": split,
                        "group": group,
                        "category": split,
                        "source_zip": "source.zip",
                        "source_path": f"{group}/image-{index}.png",
                        "local_path": "unused",
                        "width": "8",
                        "height": "8",
                    }
                )
    return rows


def test_select_group_balanced_rows_is_deterministic_and_balanced() -> None:
    first = select_group_balanced_rows(
        _source_rows(),
        per_class=4,
        reserve_per_class=0,
        seed=7,
        real_groups=("real-a", "real-b"),
        fake_groups=("fake-a", "fake-b"),
    )
    second = select_group_balanced_rows(
        _source_rows(),
        per_class=4,
        reserve_per_class=0,
        seed=7,
        real_groups=("real-a", "real-b"),
        fake_groups=("fake-a", "fake-b"),
    )

    assert first == second
    assert [row["label"] for row in first].count(0) == 4
    assert [row["label"] for row in first].count(1) == 4
    assert {row["group"] for row in first} == {
        "real-a",
        "real-b",
        "fake-a",
        "fake-b",
    }


def test_select_group_balanced_rows_rejects_forbidden_group() -> None:
    rows = _source_rows()
    rows.extend(
        {
            **rows[0],
            "split": "fake",
            "group": "DALLE3",
            "source_path": f"DALLE3/image-{index}.png",
        }
        for index in range(4)
    )

    try:
        select_group_balanced_rows(
            rows,
            per_class=2,
            reserve_per_class=0,
            seed=7,
            real_groups=("real-a",),
            fake_groups=("DALLE3",),
        )
    except ValueError as exc:
        assert "Forbidden" in str(exc)
    else:
        raise AssertionError("DALLE3 should never be accepted as a training group")


def test_assign_training_records_uses_reserve_after_hash_exclusions(
    tmp_path: Path,
) -> None:
    downloads = []
    for label, class_name in ((0, "REAL"), (1, "FAKE")):
        for rank, digest in enumerate(("frozen", "duplicate", "duplicate", "unique")):
            record = ImageRecord(
                image_id=f"{label}-{rank}",
                relative_path=f"{class_name}/{rank}.png",
                source_split="train",
                role="train",
                label=label,
                class_name=class_name,
            )
            downloads.append(
                (
                    record,
                    {
                        "selection_rank": rank,
                        "sha256": f"{label}-{digest}",
                    },
                )
            )

    records, provenance = assign_training_records(
        downloads,
        per_class=2,
        excluded_hashes={"0-frozen", "1-frozen"},
    )

    assert len(records) == 4
    assert len({row["sha256"] for row in provenance}) == 4
    assert not {"0-frozen", "1-frozen"} & {row["sha256"] for row in provenance}
