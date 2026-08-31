# Phase 4 — Inference CLI Script: Completion Record

Original goal: ship the exact required deliverable script — takes an image directory as input and outputs a
JSON file with `image_path` and `pred` for each image.

Reference: [../../problem_statement.md §5.5](../../problem_statement.md#55-expected-deliverables)

## Steps

1. `src/predict.py` — CLI accepting `--input_dir`, `--out`, and `--device auto` arguments. An
      optional `--include_label` adds a thresholded classification for demonstrations while the
      default judge-facing JSON retains the exact required `image_path` + `pred` schema.
2. Loads the selected artifact bundle and validates its schema version, backbone/checkpoint,
      feature mode and dimensions, preprocessing, class mapping, scaler, and classifier before
      prediction.
3. Iterates over all images in `--input_dir` (support common extensions: `.jpg`, `.jpeg`, `.png`,
   `.webp`), embeds each with CLIP, conditionally adds frequency features only for compatible
   legacy artifacts, and predicts a **continuous** confidence score. The promoted
   `semantic_native_mixed` artifact declares semantic mode, so FFT extraction is bypassed. The
   default output remains continuous rather than thresholded because the organizers can apply
   their own operating point on a different test distribution.
4. Writes a JSON list of `{"image_path": ..., "pred": ...}` objects to `--out`, where `pred` is the
      finite continuous AIGC confidence score. Preserve deterministic path ordering.
5. Handles empty directories, corrupt images, unsupported formats, output-directory creation, and
      per-image failures without corrupting the JSON file. Log skipped files and return a nonzero exit
      code only when no prediction can be produced.
6. Batch inference on the selected GPU but support forced CPU execution. Use the same code and
      artifact on AMD ROCm, Colab CUDA, and CPU; do not serialize device-bound tensors.

## Outcome

`src/predict.py` implements the required directory-to-JSON contract, recursive deterministic path
ordering, common raster formats, corrupt-file skipping, atomic output, configurable batching and
workers, and optional human-readable labels. Fresh GPU and CPU processes produced smoke-test
probabilities within `4.6e-9`. End-to-end runs over 100 held-out images, including model loading and
feature extraction, took 15.504 seconds on the RX 7900 XTX and 46.260 seconds on the Ryzen 7
7800X3D for the original hybrid artifact.

The promoted artifact uses the same judge-facing contract and declares semantic-only inference. Its
optional labels use the calibrated `0.781959` threshold stored in the bundle; continuous `pred`
values remain the default. This preserves compatibility with both the current 512-dimensional
classifier and the historical 544-dimensional hybrid bundle without hard-coding either path.
A fresh AMD/CPU smoke test classified `fake.jpg` as FAKE (`0.989305` / `0.989272`) and `real.jpg`
as REAL (`0.011023` / `0.010834`) with the same deterministic JSON ordering.

## Definition of done

- [x] `python src/predict.py --input_dir <folder> --out preds.json` runs successfully on a small
      held-out folder and produces valid, parseable JSON matching the exact schema (`image_path`,
      `pred`).
- [x] `--device cpu` succeeds in a fresh process with the committed artifact.
- [x] Recursive mixed-format discovery is covered independently of dataset-specific directory
      layouts.
- [x] Empty, corrupt, and mixed-format folder behavior is deterministic and documented.
- [x] Runtime is recorded for a folder of 100 images in the
      original end-to-end benchmark.
- [x] The promoted semantic artifact resolves its feature mode from metadata and skips FFT
      extraction while preserving the required JSON schema.

## Time budget

30 minutes, ending before robustness expansion begins.

## Risks

- Training and inference preprocessing drift → load configuration from the artifact bundle and fail
      clearly on a feature-dimension mismatch.
- ROCm/CUDA artifact portability → serialize CPU arrays and scikit-learn state, then smoke-test with
      `--device cpu` before recording results.
