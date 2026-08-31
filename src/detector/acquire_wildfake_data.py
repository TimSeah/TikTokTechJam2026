from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import shutil
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

import pyarrow.parquet as parquet
import requests

from src.detector.acquire_native_data import (
    create_session,
    dataset_revision,
    find_existing_image,
    inspect_image,
    read_excluded_hashes,
    read_excluded_ids,
    safe_name,
    write_json_atomic,
)
from src.detector.data import ImageRecord, write_manifest

WILDFAKE_DATASET = "buxtcodes/WildFake-Sample"
WILDFAKE_SPLIT = "train"
ROWS_PER_SHARD = 3000
SHARD_COUNT = 10
MANIFEST_URL = (
    "https://huggingface.co/datasets/{dataset}/resolve/{revision}/manifest.csv"
)
SHARD_URL = (
    "https://huggingface.co/datasets/{dataset}/resolve/{revision}/data/"
    "train-{shard:05d}-of-{shard_count:05d}.parquet"
)
FAKE_TRAIN_GROUPS = ("GALIP", "GigaGAN", "ADM", "DDPM", "VQGAN", "VQVAE")
REAL_TRAIN_GROUPS = ("afhq", "celebahq", "church")
FORBIDDEN_GROUPS = {"coco", "coco_val2017", "dalle3", "dalle3_advanced"}


def normalized_group(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def source_id(row: dict) -> str:
    basename = PurePosixPath(str(row["source_path"])).name
    return f"{row['group']}/{basename}"


def read_source_manifest(
    session: requests.Session, revision: str
) -> list[dict[str, str]]:
    response = session.get(
        MANIFEST_URL.format(dataset=WILDFAKE_DATASET, revision=revision),
        timeout=120,
    )
    response.raise_for_status()
    rows = list(csv.DictReader(io.StringIO(response.text)))
    if not rows:
        raise ValueError("WildFake source manifest is empty")
    return rows


def select_group_balanced_rows(
    source_rows: list[dict[str, str]],
    per_class: int,
    reserve_per_class: int,
    seed: int,
    excluded_ids: set[str] | None = None,
    real_groups: tuple[str, ...] = REAL_TRAIN_GROUPS,
    fake_groups: tuple[str, ...] = FAKE_TRAIN_GROUPS,
) -> list[dict]:
    if per_class < 1 or reserve_per_class < 0:
        raise ValueError("per_class must be positive and reserve_per_class non-negative")
    excluded_ids = excluded_ids or set()
    selected: list[dict] = []
    for label, split, groups in (
        (0, "real", real_groups),
        (1, "fake", fake_groups),
    ):
        forbidden = [group for group in groups if normalized_group(group) in FORBIDDEN_GROUPS]
        if forbidden:
            raise ValueError(f"Forbidden WildFake training groups requested: {forbidden}")
        target = per_class + reserve_per_class
        base_quota, remainder = divmod(target, len(groups))
        class_rows: list[dict] = []
        for group_index, group in enumerate(groups):
            candidates = []
            for global_index, row in enumerate(source_rows):
                if row["split"].lower() != split or row["group"] != group:
                    continue
                identifier = source_id(row)
                if identifier in excluded_ids or str(row["source_path"]) in excluded_ids:
                    continue
                candidates.append(
                    {
                        **row,
                        "global_index": global_index,
                        "shard_index": global_index // ROWS_PER_SHARD,
                        "shard_row_index": global_index % ROWS_PER_SHARD,
                        "label": label,
                        "class_name": "REAL" if label == 0 else "FAKE",
                        "source_id": identifier,
                    }
                )
            quota = base_quota + (group_index < remainder)
            if len(candidates) < quota:
                raise ValueError(
                    f"WildFake group {group} has {len(candidates)} candidates; need {quota}"
                )
            random_generator = random.Random(f"{seed}:{label}:{group}")
            class_rows.extend(random_generator.sample(candidates, quota))
        random.Random(f"{seed}:{label}:order").shuffle(class_rows)
        for selection_rank, row in enumerate(class_rows):
            row["selection_rank"] = selection_rank
        selected.extend(class_rows)
    return selected


def download_file_resumable(
    session: requests.Session,
    url: str,
    destination: Path,
    attempts: int = 4,
) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for _ in range(attempts):
        start = partial_path.stat().st_size if partial_path.exists() else 0
        headers = {"Range": f"bytes={start}-"} if start else {}
        try:
            with session.get(url, headers=headers, stream=True, timeout=180) as response:
                response.raise_for_status()
                append = start > 0 and response.status_code == 206
                mode = "ab" if append else "wb"
                initial_size = start if append else 0
                content_range = response.headers.get("Content-Range", "")
                range_match = re.search(r"/(\d+)$", content_range)
                expected_size = (
                    int(range_match.group(1))
                    if range_match
                    else initial_size + int(response.headers["Content-Length"])
                )
                with partial_path.open(mode) as output_file:
                    for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            output_file.write(chunk)
            if partial_path.stat().st_size != expected_size:
                raise IOError(
                    f"Incomplete download {partial_path.stat().st_size}/{expected_size}"
                )
            partial_path.replace(destination)
            return
        except (OSError, requests.RequestException) as exc:
            last_error = exc
    raise RuntimeError(f"Could not download {url}") from last_error


def extract_selected_rows(
    plan_rows: list[dict],
    shard_paths: dict[int, Path],
    output_root: Path,
    revision: str,
    seed: int,
) -> list[tuple[ImageRecord, dict]]:
    by_shard: dict[int, dict[int, dict]] = defaultdict(dict)
    for row in plan_rows:
        by_shard[int(row["shard_index"])][int(row["shard_row_index"])] = row
    downloads: list[tuple[ImageRecord, dict]] = []
    columns = [
        "image_bytes",
        "split",
        "group",
        "category",
        "source_zip",
        "source_path",
        "width",
        "height",
    ]
    for shard_index in sorted(by_shard):
        selected = by_shard[shard_index]
        row_offset = 0
        source = parquet.ParquetFile(shard_paths[shard_index])
        for batch in source.iter_batches(batch_size=32, columns=columns):
            values = batch.to_pydict()
            for batch_index in range(batch.num_rows):
                shard_row_index = row_offset + batch_index
                planned = selected.get(shard_row_index)
                if planned is None:
                    continue
                row = {name: values[name][batch_index] for name in columns}
                if row["group"] != planned["group"] or row["source_path"] != planned["source_path"]:
                    raise ValueError(
                        f"WildFake manifest and Parquet differ at row {planned['global_index']}"
                    )
                payload = bytes(row["image_bytes"])
                label = int(planned["label"])
                class_name = str(planned["class_name"])
                global_index = int(planned["global_index"])
                identifier = str(planned["source_id"])
                existing_path = find_existing_image(
                    output_root, class_name, global_index, identifier
                )
                if existing_path is not None:
                    payload = existing_path.read_bytes()
                    relative_path = existing_path.relative_to(output_root)
                    reused_existing = True
                else:
                    _, _, image_format = inspect_image(payload)
                    extension = {
                        "JPEG": ".jpg",
                        "PNG": ".png",
                        "WEBP": ".webp",
                    }.get(image_format.upper(), ".img")
                    relative_path = Path(class_name) / (
                        f"{global_index:06d}_{safe_name(identifier)}{extension}"
                    )
                    destination = output_root / relative_path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
                    temporary_path.write_bytes(payload)
                    temporary_path.replace(destination)
                    reused_existing = False
                width, height, image_format = inspect_image(payload)
                digest = hashlib.sha256(payload).hexdigest()
                record = ImageRecord(
                    image_id=f"wildfake-train-{global_index}-{safe_name(identifier)}",
                    relative_path=relative_path.as_posix(),
                    source_split=WILDFAKE_SPLIT,
                    role="train",
                    label=label,
                    class_name=class_name,
                )
                provenance = {
                    "dataset": WILDFAKE_DATASET,
                    "dataset_revision": revision,
                    "split": WILDFAKE_SPLIT,
                    "sampling_seed": seed,
                    "selection_rank": int(planned["selection_rank"]),
                    "global_index": global_index,
                    "shard_index": shard_index,
                    "shard_row_index": shard_row_index,
                    "label": label,
                    "class_name": class_name,
                    "group": row["group"],
                    "category": row["category"],
                    "source_zip": row["source_zip"],
                    "source_path": row["source_path"],
                    "source_id": identifier,
                    "width": width,
                    "height": height,
                    "format": image_format,
                    "relative_path": relative_path.as_posix(),
                    "bytes": len(payload),
                    "sha256": digest,
                    "reused_existing": reused_existing,
                    "role": "train",
                }
                downloads.append((record, provenance))
            row_offset += batch.num_rows
        print(
            f"extracted_shard={shard_index} selected={len(selected)}",
            flush=True,
        )
    return downloads


def assign_training_records(
    downloads: list[tuple[ImageRecord, dict]],
    per_class: int,
    excluded_hashes: set[str] | None = None,
) -> tuple[list[ImageRecord], list[dict]]:
    excluded_hashes = excluded_hashes or set()
    seen_hashes: set[str] = set()
    records: list[ImageRecord] = []
    provenance: list[dict] = []
    for label in (0, 1):
        candidates = sorted(
            (item for item in downloads if item[0].label == label),
            key=lambda item: int(item[1]["selection_rank"]),
        )
        selected = []
        for record, row in candidates:
            digest = str(row["sha256"]).lower()
            if digest in excluded_hashes or digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            selected.append((record, row))
            if len(selected) == per_class:
                break
        if len(selected) < per_class:
            raise RuntimeError(
                f"Only {len(selected)} unique WildFake rows for label {label}; "
                f"need {per_class}"
            )
        records.extend(record for record, _ in selected)
        provenance.extend(row for _, row in selected)
    return records, provenance


def write_provenance(rows: list[dict], destination: Path) -> None:
    if not rows:
        raise ValueError("Cannot write empty provenance")
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(destination)


def download_wildfake_train(
    output_root: Path,
    per_class: int,
    seed: int,
    exclude_manifests: Iterable[Path] = (),
    reserve_per_class: int = 100,
    keep_shards: bool = False,
) -> Path:
    session = create_session()
    revision = dataset_revision(session, WILDFAKE_DATASET)
    excluded_ids = read_excluded_ids(exclude_manifests)
    excluded_hashes = read_excluded_hashes(exclude_manifests)
    exclusion_digest = hashlib.sha256(
        "\n".join(sorted(excluded_ids | excluded_hashes)).encode("utf-8")
    ).hexdigest()
    plan_config = {
        "dataset": WILDFAKE_DATASET,
        "dataset_revision": revision,
        "split": WILDFAKE_SPLIT,
        "seed": seed,
        "per_class": per_class,
        "reserve_per_class": reserve_per_class,
        "real_groups": list(REAL_TRAIN_GROUPS),
        "fake_groups": list(FAKE_TRAIN_GROUPS),
        "forbidden_groups": sorted(FORBIDDEN_GROUPS),
        "exclusion_digest": exclusion_digest,
    }
    plan_path = output_root / "download_plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("config") != plan_config:
            raise ValueError(f"Existing WildFake plan does not match: {plan_path}")
        selected = plan["rows"]
        print(f"resuming_plan={plan_path} candidates={len(selected)}", flush=True)
    else:
        source_rows = read_source_manifest(session, revision)
        selected = select_group_balanced_rows(
            source_rows,
            per_class,
            reserve_per_class,
            seed,
            excluded_ids,
        )
        write_json_atomic({"config": plan_config, "rows": selected}, plan_path)
        print(f"created_plan={plan_path} candidates={len(selected)}", flush=True)

    shard_dir = output_root / ".shards"
    shard_paths: dict[int, Path] = {}
    for shard_index in sorted({int(row["shard_index"]) for row in selected}):
        shard_path = shard_dir / f"train-{shard_index:05d}-of-{SHARD_COUNT:05d}.parquet"
        url = SHARD_URL.format(
            dataset=WILDFAKE_DATASET,
            revision=revision,
            shard=shard_index,
            shard_count=SHARD_COUNT,
        )
        print(f"downloading_shard={shard_index} destination={shard_path}", flush=True)
        download_file_resumable(session, url, shard_path)
        shard_paths[shard_index] = shard_path

    downloads = extract_selected_rows(selected, shard_paths, output_root, revision, seed)
    records, provenance = assign_training_records(
        downloads, per_class, excluded_hashes
    )
    manifest_path = output_root / "manifest-train.csv"
    write_manifest(records, manifest_path)
    write_provenance(provenance, output_root / "provenance.csv")
    write_json_atomic(
        {
            **plan_config,
            "complete": True,
            "train_real": sum(record.label == 0 for record in records),
            "train_fake": sum(record.label == 1 for record in records),
            "selected_groups": sorted({row["group"] for row in provenance}),
            "excluded_ids": len(excluded_ids),
            "excluded_hashes": len(excluded_hashes),
        },
        output_root / "metadata.json",
    )
    if not keep_shards:
        shutil.rmtree(shard_dir)
    print(
        f"dataset={WILDFAKE_DATASET} revision={revision} "
        f"train_real={per_class} train_fake={per_class} manifest={manifest_path}",
        flush=True,
    )
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a legal, source-diverse WildFake training sample."
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/native-train/wildfake-final")
    )
    parser.add_argument("--per-class", type=int, default=4000)
    parser.add_argument("--reserve-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--keep-shards", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_wildfake_train(
        args.output_root,
        args.per_class,
        args.seed,
        args.exclude_manifest,
        args.reserve_per_class,
        args.keep_shards,
    )


if __name__ == "__main__":
    main()