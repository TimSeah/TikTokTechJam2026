# Phase 4 — Inference CLI Script

Goal: ship the exact required deliverable script — takes an image directory as input and outputs a
JSON file with `image_path` and `pred` for each image.

Reference: [../../problem_statement.md §5.5](../../problem_statement.md#55-expected-deliverables)

## Steps

1. `src/predict.py` — CLI accepting `--input_dir` and `--out` arguments.
2. Loads the frozen CLIP model, the frequency-feature extractor, and the trained probe head from
   the Phase 2 artifacts.
3. Iterates over all images in `--input_dir` (support common extensions: `.jpg`, `.jpeg`, `.png`,
   `.webp`), embeds each (CLIP + frequency features, matching Phase 2), and predicts a
   **continuous** confidence score — do not threshold to a hard label. The organizers apply their
   own continuous threshold during judging since the test distribution differs from the training
   distribution.
4. Writes a JSON list of `{"image_path": ..., "pred": ...}` objects to `--out`, where `pred` is the
   continuous confidence score.
5. Handles basic errors gracefully (corrupt image, unsupported format) — skip with a logged
   warning rather than crashing the whole run.

## Definition of done

- [ ] `python src/predict.py --input_dir <folder> --out preds.json` runs successfully on a small
      held-out folder and produces valid, parseable JSON matching the exact schema (`image_path`,
      `pred`).
- [ ] Confirmed to work on a folder containing both CIFAKE-style and WildFake-style images without
      crashing.
- [ ] Runtime is reasonable for a folder of ~50–100 images (document the observed runtime in the
      README).

## Time budget

45 minutes.

## Risks

None major — this is the most mechanical phase. Keep the script simple; do not add speculative
CLI flags beyond `--input_dir` and `--out`.
