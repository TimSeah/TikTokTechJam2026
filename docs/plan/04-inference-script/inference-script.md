# Phase 4 — Inference CLI Script

Goal: ship the exact required deliverable script — takes an image directory as input and outputs a
JSON file with `image_path` and `pred` for each image.

Reference: [../../problem_statement.md §5.5](../../problem_statement.md#55-expected-deliverables)

## Steps

1. `src/predict.py` — CLI accepting `--input_dir`, `--out`, and `--device auto` arguments. An
      optional `--include_label` adds a thresholded classification for demonstrations while the
      default judge-facing JSON retains the exact required `image_path` + `pred` schema.
2. Loads the Phase 2 artifact bundle and validates the backbone/checkpoint, feature dimensions,
      preprocessing, frequency configuration, class mapping, and scaler before prediction.
3. Iterates over all images in `--input_dir` (support common extensions: `.jpg`, `.jpeg`, `.png`,
   `.webp`), embeds each (CLIP + frequency features, matching Phase 2), and predicts a
   **continuous** confidence score — do not threshold to a hard label. The organizers apply their
   own continuous threshold during judging since the test distribution differs from the training
   distribution.
4. Writes a JSON list of `{"image_path": ..., "pred": ...}` objects to `--out`, where `pred` is the
      finite continuous AIGC confidence score. Preserve deterministic path ordering.
5. Handles empty directories, corrupt images, unsupported formats, output-directory creation, and
      per-image failures without corrupting the JSON file. Log skipped files and return a nonzero exit
      code only when no prediction can be produced.
6. Batch inference on the selected GPU but support forced CPU execution. Use the same code and
      artifact on AMD ROCm, Colab CUDA, and CPU; do not serialize device-bound tensors.

## Definition of done

- [ ] `python src/predict.py --input_dir <folder> --out preds.json` runs successfully on a small
      held-out folder and produces valid, parseable JSON matching the exact schema (`image_path`,
      `pred`).
- [ ] `--device cpu` succeeds in a fresh process with the committed artifact.
- [ ] Confirmed to work on a folder containing both CIFAKE-style and WildFake-style images without
      crashing.
- [ ] Empty, corrupt, and mixed-format folder smoke tests have deterministic documented behavior.
- [ ] Runtime is reasonable for a folder of ~50–100 images (document the observed runtime in the
      README).

## Time budget

30 minutes, ending before robustness expansion begins.

## Risks

- Training and inference preprocessing drift → load configuration from the artifact bundle and fail
      clearly on a feature-dimension mismatch.
- ROCm/CUDA artifact portability → serialize CPU arrays and scikit-learn state, then smoke-test with
      `--device cpu` before recording results.
