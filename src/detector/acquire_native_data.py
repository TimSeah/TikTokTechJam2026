from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from functools import partial
from pathlib import Path
from threading import local

import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.detector.data import ImageRecord, write_manifest

DATASET_API = "https://huggingface.co/api/datasets"
ROWS_API = "https://datasets-server.huggingface.co/rows"
SID_DATASET = "saberzl/SID_Set"
SID_CONFIG = "default"
SID_SPLIT = "train"
PAGE_SIZE = 100
DEFAULT_WORKERS = 8
DEFAULT_PAGE_WORKERS = 1
DEFAULT_RETRY_ROUNDS = 3
_THREAD_STATE = local()


def create_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=8,
        connect=8,
        read=8,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def worker_session() -> requests.Session:
    session = getattr(_THREAD_STATE, "session", None)
    if session is None:
        session = create_session()
        _THREAD_STATE.session = session
    return session


def dataset_revision(session: requests.Session, dataset: str) -> str:
    response = session.get(f"{DATASET_API}/{dataset}", timeout=60)
    response.raise_for_status()
    revision = response.json().get("sha")
    if not revision:
        raise ValueError(f"Dataset metadata has no revision: {dataset}")
    return str(revision)


def fetch_page(
    session: requests.Session,
    dataset: str,
    config: str,
    split: str,
    offset: int,
) -> dict:
    response = session.get(
        ROWS_API,
        params={
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": PAGE_SIZE,
        },
        timeout=90,
    )
    response.raise_for_status()
    return response.json()


def fetch_pages(
    offsets: list[int],
    first_page: dict,
    workers: int,
    cache_dir: Path | None = None,
) -> Iterable[dict]:
    def fetch_offset(offset: int) -> dict:
        cache_path = cache_dir / f"page-{offset:09d}.json" if cache_dir else None
        if cache_path is not None and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        page = (
            first_page
            if offset == 0
            else fetch_page(
                worker_session(), SID_DATASET, SID_CONFIG, SID_SPLIT, offset
            )
        )
        if cache_path is not None:
            write_json_atomic(page, cache_path)
        return page

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, page in enumerate(executor.map(fetch_offset, offsets), start=1):
            if index % 10 == 0 or index == len(offsets):
                print(f"selection_pages={index}/{len(offsets)}", flush=True)
            yield page


def select_balanced_rows(
    pages: Iterable[dict],
    per_class: int,
    seed: int,
    excluded_ids: set[str] | None = None,
) -> list[dict]:
    if per_class < 1:
        raise ValueError("per_class must be positive")
    random_generator = random.Random(seed)
    excluded_ids = excluded_ids or set()
    selected: dict[int, list[dict]] = {0: [], 1: []}
    for page in pages:
        rows = list(page["rows"])
        random_generator.shuffle(rows)
        for wrapped in rows:
            row = wrapped["row"]
            label = int(row["label"])
            if str(row.get("img_id", "")) in excluded_ids:
                continue
            if label in selected and len(selected[label]) < per_class:
                selected[label].append(wrapped)
        if all(
            len(rows_for_class) == per_class for rows_for_class in selected.values()
        ):
            break
    counts = {label: len(rows_for_class) for label, rows_for_class in selected.items()}
    if any(count != per_class for count in counts.values()):
        raise ValueError(f"Insufficient balanced rows: {counts}")
    return selected[0] + selected[1]


def safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return normalized[-100:] or "image"


def read_excluded_ids(manifest_paths: Iterable[Path]) -> set[str]:
    excluded_ids: set[str] = set()
    for manifest_path in manifest_paths:
        with manifest_path.open("r", encoding="utf-8", newline="") as manifest_file:
            for row in csv.DictReader(manifest_file):
                identifier = row.get("img_id") or row.get("item_id")
                if identifier:
                    excluded_ids.add(identifier)
    return excluded_ids


def read_excluded_hashes(manifest_paths: Iterable[Path]) -> set[str]:
    excluded_hashes: set[str] = set()
    for manifest_path in manifest_paths:
        with manifest_path.open("r", encoding="utf-8", newline="") as manifest_file:
            for row in csv.DictReader(manifest_file):
                digest = row.get("sha256")
                if digest:
                    excluded_hashes.add(digest.lower())
    return excluded_hashes


def write_json_atomic(value: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(destination)


def inspect_image(payload: bytes) -> tuple[int, int, str]:
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        width, height = image.size
        image_format = image.format or "UNKNOWN"
    return width, height, image_format


def find_existing_image(
    output_root: Path, class_name: str, row_index: int, image_id: str
) -> Path | None:
    prefix = f"{row_index:06d}_{safe_name(image_id)}."
    candidates = [
        path
        for path in (output_root / class_name).glob(f"{prefix}*")
        if not path.name.endswith(".tmp")
    ]
    if len(candidates) > 1:
        raise ValueError(f"Multiple existing images found for row {row_index}")
    return candidates[0] if candidates else None


def download_selected_row(
    wrapped: dict, output_root: Path, revision: str, seed: int
) -> tuple[ImageRecord, dict]:
    row_index = int(wrapped["row_idx"])
    row = wrapped["row"]
    label = int(row["label"])
    class_name = "REAL" if label == 0 else "FAKE"
    image_id = str(row["img_id"])
    existing_path = find_existing_image(output_root, class_name, row_index, image_id)
    if existing_path is not None:
        payload = existing_path.read_bytes()
        width, height, image_format = inspect_image(payload)
        relative_path = existing_path.relative_to(output_root)
        reused_existing = True
    else:
        image_response = worker_session().get(row["image"]["src"], timeout=180)
        image_response.raise_for_status()
        payload = image_response.content
        width, height, image_format = inspect_image(payload)
        extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(
            image_format.upper(), ".img"
        )
        relative_path = (
            Path(class_name) / f"{row_index:06d}_{safe_name(image_id)}{extension}"
        )
        destination = output_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = destination.with_suffix(destination.suffix + ".tmp")
        temporary_path.write_bytes(payload)
        temporary_path.replace(destination)
        reused_existing = False

    digest = hashlib.sha256(payload).hexdigest()

    record = ImageRecord(
        image_id=f"sid-train-{row_index}-{image_id}",
        relative_path=relative_path.as_posix(),
        source_split=SID_SPLIT,
        role="train",
        label=label,
        class_name=class_name,
    )
    provenance = {
        "dataset": SID_DATASET,
        "dataset_revision": revision,
        "config": SID_CONFIG,
        "split": SID_SPLIT,
        "sampling_seed": seed,
        "row_idx": row_index,
        "img_id": image_id,
        "label": label,
        "class_name": class_name,
        "width": width,
        "height": height,
        "format": image_format,
        "relative_path": relative_path.as_posix(),
        "bytes": len(payload),
        "sha256": digest,
        "reused_existing": reused_existing,
    }
    return record, provenance


def assign_split_roles(
    downloads: list[tuple[ImageRecord, dict]],
    train_per_class: int,
    eval_per_class: int,
    excluded_hashes: set[str] | None = None,
) -> tuple[list[ImageRecord], list[ImageRecord], list[ImageRecord], list[dict]]:
    required_per_class = train_per_class + eval_per_class
    excluded_hashes = excluded_hashes or set()
    seen_hashes: set[str] = set()
    unique_downloads: list[tuple[ImageRecord, dict]] = []
    for item in downloads:
        digest = str(item[1].get("sha256", "")).lower()
        if digest and (digest in excluded_hashes or digest in seen_hashes):
            continue
        if digest:
            seen_hashes.add(digest)
        unique_downloads.append(item)
    all_records: list[ImageRecord] = []
    train_records: list[ImageRecord] = []
    eval_records: list[ImageRecord] = []
    provenance_rows: list[dict] = []
    for label in (0, 1):
        class_downloads = [item for item in unique_downloads if item[0].label == label]
        if len(class_downloads) < required_per_class:
            raise RuntimeError(
                f"Only {len(class_downloads)} successful downloads for label {label}; "
                f"need {required_per_class}"
            )
        for rank, (record, provenance) in enumerate(
            class_downloads[:required_per_class]
        ):
            role = "train" if rank < train_per_class else "evaluation"
            assigned_record = replace(record, role=role)
            assigned_provenance = {**provenance, "role": role}
            all_records.append(assigned_record)
            provenance_rows.append(assigned_provenance)
            if role == "train":
                train_records.append(assigned_record)
            else:
                eval_records.append(assigned_record)
    return all_records, train_records, eval_records, provenance_rows


def download_sid_train(
    output_root: Path,
    per_class: int,
    seed: int,
    max_pages: int | None = None,
    exclude_manifests: Iterable[Path] = (),
    workers: int = DEFAULT_WORKERS,
    eval_per_class: int = 0,
    reserve_per_class: int = 0,
    retry_rounds: int = DEFAULT_RETRY_ROUNDS,
    page_workers: int = DEFAULT_PAGE_WORKERS,
) -> Path:
    if workers < 1:
        raise ValueError("workers must be positive")
    if page_workers < 1:
        raise ValueError("page_workers must be positive")
    if retry_rounds < 1:
        raise ValueError("retry_rounds must be positive")
    if eval_per_class < 0 or reserve_per_class < 0:
        raise ValueError("eval_per_class and reserve_per_class cannot be negative")
    session = create_session()
    revision = dataset_revision(session, SID_DATASET)
    excluded_ids = read_excluded_ids(exclude_manifests)
    excluded_hashes = read_excluded_hashes(exclude_manifests)
    excluded_ids_sha256 = hashlib.sha256(
        "\n".join(sorted(excluded_ids)).encode("utf-8")
    ).hexdigest()
    candidate_per_class = per_class + eval_per_class + reserve_per_class
    plan_config = {
        "dataset": SID_DATASET,
        "dataset_revision": revision,
        "config": SID_CONFIG,
        "split": SID_SPLIT,
        "seed": seed,
        "candidate_per_class": candidate_per_class,
        "excluded_ids_sha256": excluded_ids_sha256,
        "max_pages": max_pages,
    }
    plan_path = output_root / "download_plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("config") != plan_config:
            raise ValueError(
                f"Existing download plan does not match this run: {plan_path}"
            )
        selected = plan["rows"]
        print(f"resuming_plan={plan_path} candidates={len(selected)}", flush=True)
    else:
        page_cache_dir = output_root / ".selection-pages" / revision
        first_page_path = page_cache_dir / "page-000000000.json"
        if first_page_path.exists():
            first_page = json.loads(first_page_path.read_text(encoding="utf-8"))
        else:
            first_page = fetch_page(session, SID_DATASET, SID_CONFIG, SID_SPLIT, 0)
            write_json_atomic(first_page, first_page_path)
        total_rows = int(first_page["num_rows_total"])
        offsets = list(range(0, total_rows, PAGE_SIZE))
        random.Random(seed).shuffle(offsets)
        if max_pages is not None:
            offsets = offsets[:max_pages]
        pages = fetch_pages(offsets, first_page, page_workers, page_cache_dir)
        selected = select_balanced_rows(
            pages, candidate_per_class, seed, excluded_ids
        )
        write_json_atomic({"config": plan_config, "rows": selected}, plan_path)
        print(f"created_plan={plan_path} candidates={len(selected)}", flush=True)

    download_row = partial(
        download_selected_row, output_root=output_root, revision=revision, seed=seed
    )
    successful: dict[int, tuple[ImageRecord, dict]] = {}
    pending = {index: wrapped for index, wrapped in enumerate(selected)}
    failure_history: list[dict] = []
    progress_path = output_root / "download_progress.json"
    for attempt in range(1, retry_rounds + 1):
        failed: dict[int, dict] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(download_row, wrapped): (index, wrapped)
                for index, wrapped in pending.items()
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                index, wrapped = futures[future]
                try:
                    successful[index] = future.result()
                except Exception as exc:
                    row = wrapped["row"]
                    error = {
                        "attempt": attempt,
                        "plan_index": index,
                        "row_idx": wrapped.get("row_idx"),
                        "img_id": row.get("img_id"),
                        "label": row.get("label"),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    failure_history.append(error)
                    failed[index] = wrapped
                    print(
                        f"download_failed attempt={attempt} row={error['row_idx']} "
                        f"label={error['label']} error={error['error']}",
                        flush=True,
                    )
                if completed % 100 == 0 or completed == len(futures):
                    write_json_atomic(
                        {
                            "complete": False,
                            "attempt": attempt,
                            "total_candidates": len(selected),
                            "successful_plan_indexes": sorted(successful),
                            "failure_history": failure_history,
                        },
                        progress_path,
                    )
                    print(
                        f"attempt={attempt} completed={completed}/{len(futures)} "
                        f"successful={len(successful)} failed={len(failed)}",
                        flush=True,
                    )
        pending = failed
        if not pending:
            break

    ordered_downloads = [successful[index] for index in sorted(successful)]
    records, train_records, eval_records, provenance_rows = assign_split_roles(
        ordered_downloads, per_class, eval_per_class, excluded_hashes
    )

    manifest_path = output_root / "manifest.csv"
    write_manifest(records, manifest_path)
    train_manifest_path = output_root / "manifest-train.csv"
    write_manifest(train_records, train_manifest_path)
    if eval_records:
        write_manifest(eval_records, output_root / "manifest-eval.csv")
    provenance_path = output_root / "provenance.csv"
    temporary_path = provenance_path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(provenance_rows[0]))
        writer.writeheader()
        writer.writerows(provenance_rows)
    temporary_path.replace(provenance_path)
    final_errors = [
        error for error in failure_history if error["plan_index"] in pending
    ]
    write_json_atomic(
        {
            "complete": True,
            "attempts": min(retry_rounds, max(1, failure_history[-1]["attempt"]))
            if failure_history
            else 1,
            "total_candidates": len(selected),
            "successful_plan_indexes": sorted(successful),
            "failed_plan_indexes": sorted(pending),
            "failure_history": failure_history,
        },
        progress_path,
    )
    if final_errors:
        write_json_atomic({"errors": final_errors}, output_root / "download_errors.json")
    print(
        f"dataset={SID_DATASET} revision={revision} split={SID_SPLIT} "
        f"train_real={per_class} train_fake={per_class} "
        f"eval_real={eval_per_class} eval_fake={eval_per_class} "
        f"failed_candidates={len(final_errors)} excluded_ids={len(excluded_ids)} "
        f"excluded_hashes={len(excluded_hashes)} "
        f"train_manifest={train_manifest_path}"
    )
    return train_manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a deterministic balanced SID_Set training sample."
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/native-train/sid")
    )
    parser.add_argument("--per-class", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--page-workers", type=int, default=DEFAULT_PAGE_WORKERS)
    parser.add_argument("--eval-per-class", type=int, default=500)
    parser.add_argument("--reserve-per-class", type=int, default=100)
    parser.add_argument("--retry-rounds", type=int, default=DEFAULT_RETRY_ROUNDS)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_sid_train(
        args.output_root,
        args.per_class,
        args.seed,
        args.max_pages,
        args.exclude_manifest,
        args.workers,
        args.eval_per_class,
        args.reserve_per_class,
        args.retry_rounds,
        args.page_workers,
    )


if __name__ == "__main__":
    main()
