# Phase 5 — Error Analysis: Completion Record

Original goal: produce the required **Error Analysis Note** — concrete false positives/negatives with a
stated hypothesis, informed by Phase 3's robustness results.

Reference: [../../problem_statement.md §5.5](../../problem_statement.md#55-expected-deliverables)

## Steps

1. Run `predict.py` (or reuse Phase 3 eval outputs) over the CIFAKE test split and the
   validation-only subset; identify misclassified examples.
2. Select at least 2 representative false positives and 2 false negatives, prioritizing examples
   tied to the largest clean-to-transformed AUC drop found in Phase 3.
3. For each, save the image (or a thumbnail) into `outputs/error_examples/` and write one or two
   sentences hypothesizing why it was misclassified.
4. Summarize overall trade-offs observed (e.g. "aggressive JPEG compression pushes fakes below the
   decision threshold; heavily color-jittered real photos get flagged as fake").
5. Interpret the Phase 3 ablation: state whether frequency fusion and augmentation improved clean
   AUC, robust AUC, both, or neither. Do not claim a contribution that the table does not support.

## Outcome

[outputs/error_analysis.md](../../../outputs/error_analysis.md) documents two high-confidence false
positives and two high-confidence false negatives for the promoted `semantic_native_mixed` model
under severe blur, with the four transformed images and exact metadata in
`outputs/error_examples/`. The export uses the artifact's calibrated `0.781959` threshold. It
connects the observed smoothing failures to the current per-condition table while keeping visual
hypotheses separate from measured results.

The current sweep identifies quarter-scale resizing as the weakest condition (`0.774426` AUC),
followed by blur sigma 2.0 (`0.785540`) and noise sigma 0.10 (`0.797451`). Severe blur was selected
for the image-backed analysis because it is both a high-impact failure and straightforward to
inspect visually. Historical FFT ablations remain in the discussion only to explain why the
frequency branch was removed from the promoted model.

## Definition of done

- [x] `outputs/error_analysis.md` exists with at least 2 false-positive and 2 false-negative
   examples, each with an image reference and a one-line hypothesis.
- [x] A short trade-offs paragraph ties the findings back to the robustness table from Phase 3.
- [x] A short ablation paragraph distinguishes measured findings from hypotheses.

## Time budget

45 minutes including ablation interpretation.
