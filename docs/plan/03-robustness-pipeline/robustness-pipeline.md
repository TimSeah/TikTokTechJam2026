# Phase 3 — Robustness Pipeline: Completion Record

Original goal: evaluate the final detector and its ablations against all 6 transform families, expand to all
14 listed transform/severity conditions when the measured GPU throughput permits, and report the
competition's composite score with an explicit aggregation definition.

Reference: [../../problem_statement.md §5.2](../../problem_statement.md#52-problem-statement),
[../../problem_statement.md §5.7](../../problem_statement.md#57-technical-workshop-notes-supplementary)
(Slide 9: evaluation strategy and scoring formula).

## Steps

1. Reuse `src/detector/transforms.py` and the held-out manifest from Phase 2. Apply each transform
   **individually** per image (never combined), matching the competition's stated test protocol:
   - JPEG compression: quality ∈ {90, 70, 50, 30}
   - Gaussian blur: kernel σ ∈ {0.5, 1.0, 2.0}
   - Resize: scale ∈ {0.5×, 0.25×}, then upscale back
   - Gaussian noise: σ ∈ {0.02, 0.05, 0.10}
   - Color jitter: brightness/contrast/saturation ±20%
   - Center crop: 80%
2. Establish minimum breadth first using JPEG 50, blur 1.0, resize 0.5, noise 0.05, jitter 20%, and
   crop 80%. Only after all six rows exist should evaluation add the remaining eight severities.
3. `src/detector/evaluate.py` — cache semantic and frequency features once per condition in
   restartable shards. Score all three original Phase 2 ablation models from those same caches
   without repeating backbone inference. For the promoted semantic artifact, resolve feature mode
   from its configuration and bypass the cached frequency arrays.
4. Compute `AUC_clean`, accuracy, and F1 for each model and condition. Keep AUC as the primary
   threshold-free metric; accuracy and F1 use a documented threshold only for interpretation.
5. Because the workshop does not define how severities are aggregated, report both:
   - condition-weighted robust AUC: mean of every evaluated transform/severity row;
   - family-balanced robust AUC: mean within each family, then mean across the 6 families.
6. Use the condition-weighted robust AUC for the headline
   `Final Score = 0.50 × AUC_clean + 0.50 × AUC_robust`, and show the family-balanced score beside
   it as a sensitivity check. Record the exact evaluated condition list in the result metadata.
7. If the exact WildFake validation-only subset is available after required evaluation completes,
   compute clean cross-generator AUC for COCO val2017 vs. DALL·E Advanced. Do not spend the core
   evaluation window running a full transformed WildFake sweep, and verify path/hash non-overlap
   with the training manifest.
8. Write `outputs/robustness_table.csv`, `outputs/ablation_table.csv`, and a compact Markdown table
   for the README. Flag the largest absolute and relative AUC drops for Phase 5.

## Outcome

All 14 planned conditions completed on the same 20,000-image held-out split. In the original Phase
3 result, `hybrid_augmented` achieved clean AUC `0.988409`, condition-weighted robust AUC
`0.956211`, family-balanced robust AUC `0.957598`, and Final Score `0.972310`.

After this model-selection result was frozen, balanced SID and WildFake diagnostics showed that the
submitted hybrid ranked at chance cross-domain while its semantic-only component retained AUC
between `0.740600` and `0.886975`. That failure triggered semantic-only, multi-domain retraining and
made native evaluation a mandatory promotion gate.

The current `semantic_native_mixed` sweep is stored in
[outputs/robustness_table.csv](../../../outputs/robustness_table.csv) and
[outputs/ablation_table.csv](../../../outputs/ablation_table.csv). It achieves clean AUC `0.957820`,
condition-weighted robust AUC `0.896636`, family-balanced robust AUC `0.900314`, and Final Score
`0.927228`. Quarter-scale resizing (`0.774426`), blur sigma 2.0 (`0.785540`), and noise sigma 0.10
(`0.797451`) are its weakest conditions. The lower CIFAKE score is accepted because the promoted
model also passes SID validation (`0.991900`) and both WildFake gates (`0.902925`, `0.912700`), with
non-collapsed class rates and score distributions. See the
[promotion metrics](../../../outputs/native_metrics.json) and
[original-model diagnosis](../../../outputs/cross_domain_summary.json).

## Definition of done

- [x] `AUC_clean`, `AUC_robust` (averaged across all evaluated transform/severity combinations), and
      the composite `Final Score = 0.5×AUC_clean + 0.5×AUC_robust` are computed and recorded.
- [x] Every transform family has at least one measured condition; all 14 listed conditions are
   included.
- [x] Both condition-weighted and family-balanced robust AUC are reported with the exact aggregation
   definition and evaluated condition list.
- [x] Current robustness and ablation tables contain the promoted model's clean baseline and every
   completed CIFAKE transform/severity condition; the versioned diagnostic summary preserves the
   original three-model ablation.
- [x] The full WildFake reference was unavailable within its optional gate; later 400-image
   diagnostics are reported separately rather than relabeled as the full benchmark.
- [x] A Markdown-rendered summary, including the current Final Score and historical comparison, is
   present in the README.
- [x] The two weakest conditions and the measured augmentation trade-off are documented.

## Time budget

1 hour 30 minutes, ending at the 6:00 hard cut. Feature extraction is the expensive step; all model
ablations reuse the same cached condition features.

## Risks

- Full severity sweep too slow → preserve one representative condition from every family, stop at
   6:00, and label the headline value as a reduced-protocol score with its condition list.
- Colab/runtime interruption → resume from per-condition shards and calculate tables from completed
   conditions rather than rerunning successful backbone passes.
- AUC aggregation ambiguity → publish both condition-weighted and family-balanced values instead of
   implying the organizer specified one interpretation.
