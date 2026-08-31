# Phase 3 — Robustness Pipeline

Goal: evaluate the Phase 2 detector against the 6 required transform families and report the
competition's own composite scoring formula — the most directly-judged artifact besides the code
itself.

Reference: [../../problem_statement.md §5.2](../../problem_statement.md#52-problem-statement),
[../../problem_statement.md §5.7](../../problem_statement.md#57-technical-workshop-notes-supplementary)
(Slide 9: evaluation strategy and scoring formula).

## Steps

1. Reuse `src/detector/transforms.py` (built in Phase 2) — no new transform code needed here. Apply
   each transform **individually** per image (never combined), matching the competition's stated
   test protocol:
   - JPEG compression: quality ∈ {90, 70, 50, 30}
   - Gaussian blur: kernel σ ∈ {0.5, 1.0, 2.0}
   - Resize: scale ∈ {0.5×, 0.25×}, then upscale back
   - Gaussian noise: σ ∈ {0.02, 0.05, 0.10}
   - Color jitter: brightness/contrast/saturation ±20%
   - Center crop: 80%
2. `src/detector/evaluate.py` — for the CIFAKE clean test split, compute `AUC_clean`. Then, for
   each transform family/severity, apply it to the test split, re-embed (CLIP + frequency
   features), re-predict, and compute AUC per transform/severity; average across all
   transform/severity combinations to get `AUC_robust`.
3. Compute the composite score exactly as specified in the workshop slides:
   `Final Score = 0.50 × AUC_clean + 0.50 × AUC_robust`.
4. Separately, run the same clean-vs-transformed evaluation on the validation-only WildFake subset
   (COCO val2017 vs. DALL·E Advanced) and report its AUC as the **cross-generator /
   unseen-generator** generalization score — "the real generalization test" per Slide 9. This
   dataset is for evaluation only; confirm again it was never used in Phase 2 training.
5. Aggregate everything into `outputs/robustness_table.csv` (per transform/severity AUC + accuracy)
   plus a rendered markdown table for the README/report, headlined by the Final Score and the
   cross-generator AUC.
6. Flag the largest AUC drops — these feed directly into Phase 5 (error analysis).

## Definition of done

- [ ] `AUC_clean`, `AUC_robust` (averaged across all evaluated transform/severity combinations), and
      the composite `Final Score = 0.5×AUC_clean + 0.5×AUC_robust` are computed and recorded.
- [ ] A separate cross-generator/unseen-generator AUC is computed from the validation-only subset.
- [ ] `outputs/robustness_table.csv` contains the clean baseline plus every transform/severity
      combination evaluated, for both the CIFAKE test split and the validation-only subset.
- [ ] A markdown-rendered version of the table (including the headline Final Score) is ready to
      paste into the README/report.
- [ ] At least one clear qualitative finding is written down (e.g. "AUC drops most under transform
      X at severity Y").

## Time budget

45 minutes (shrank from 1h — transform implementation moved to Phase 2, so this phase is
evaluation-only).

## Risks

- Full severity sweep too slow → cover at minimum the mildest and harshest severity per transform
  rather than all listed values, and document the reduced sweep as a scoping decision in the
  README's limitations section.
