# Phase 3 — Robustness Pipeline

Goal: evaluate the final detector and its ablations against all 6 transform families, expand to all
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
   restartable shards. Score all three Phase 2 ablation models from those same caches without
   repeating backbone inference.
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

## Definition of done

- [ ] `AUC_clean`, `AUC_robust` (averaged across all evaluated transform/severity combinations), and
      the composite `Final Score = 0.5×AUC_clean + 0.5×AUC_robust` are computed and recorded.
- [ ] Every transform family has at least one measured condition; all 14 listed conditions are
   included when they finish before the 6:00 hard cut.
- [ ] Both condition-weighted and family-balanced robust AUC are reported with the exact aggregation
   definition and evaluated condition list.
- [ ] `outputs/robustness_table.csv` contains the clean baseline and every completed CIFAKE
   transform/severity condition; `outputs/ablation_table.csv` compares all three models.
- [ ] If the exact validation-only subset is available, a separate clean cross-generator AUC is
   recorded and data non-overlap is confirmed.
- [ ] A markdown-rendered version of the table (including the headline Final Score) is ready to
      paste into the README/report.
- [ ] At least one clear qualitative finding is written down (e.g. "AUC drops most under transform
      X at severity Y").

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
