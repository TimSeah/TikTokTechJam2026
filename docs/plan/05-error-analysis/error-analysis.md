# Phase 5 — Error Analysis

Goal: produce the required **Error Analysis Note** — concrete false positives/negatives with a
stated hypothesis, informed by Phase 3's robustness results.

Reference: [../../problem_statement.md §5.5](../../problem_statement.md#55-expected-deliverables)

## Steps

1. Run `predict.py` (or reuse Phase 3 eval outputs) over the CIFAKE test split and the
   validation-only subset; identify misclassified examples.
2. Select 4–8 representative false positives and false negatives, prioritizing ones tied to the
   worst-performing transform/severity found in Phase 3.
3. For each, save the image (or a thumbnail) into `outputs/error_examples/` and write one or two
   sentences hypothesizing why it was misclassified.
4. Summarize overall trade-offs observed (e.g. "aggressive JPEG compression pushes fakes below the
   decision threshold; heavily color-jittered real photos get flagged as fake").

## Definition of done

- [ ] `outputs/error_analysis.md` exists with at least 4 false-positive and 4 false-negative
      examples, each with an image reference and a one-line hypothesis.
- [ ] A short trade-offs paragraph ties the findings back to the robustness table from Phase 3.

## Time budget

35 minutes.
