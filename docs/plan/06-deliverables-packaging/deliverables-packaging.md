# Phase 6 — Deliverables Packaging

Goal: assemble and submit all required Track 5 deliverables.

Reference: [../../problem_statement.md §5.5](../../problem_statement.md#55-expected-deliverables)

## Steps

1. Write `outputs/trade_offs.md`: a short, explicit discussion of robustness vs. clean accuracy,
   generalization vs. specialization, and complexity vs. feasibility — the three trade-offs the
   workshop slides ask judges to look for (Slide 10).
2. Write the top-level `README.md`: project overview, setup/install instructions, reproduction
   steps for native AMD, Colab/CUDA, and CPU inference (exact commands per phase), limitations
   (be explicit — e.g. CIFAKE is low-resolution
   32×32, reduced severity/augmentation sweep if scope was cut), an explicit **"why this isn't a
   direct replication"** note describing the hybrid CLIP + frequency-domain design and the
   augmented-training recipe, the ablation evidence, confirmation that the LICENSE (MIT/Apache) is
   present, dataset attribution, and team contributions.
3. Write the Devpost "Written Project Description": problem framing, approach (hybrid CLIP +
   frequency-domain detector trained with augmentation), tools/APIs/libraries/datasets used, and the
   headline Final Score + cross-generator AUC from Phase 3.
4. Record a 2–4 minute demo video: show CPU-capable `predict.py` running on a folder, inspect the
   output JSON, then walk through the headline robustness and ablation tables plus 1–2 error
   examples. Show cross-generator results only if the exact validation subset was used. Upload to
   YouTube as public, and link it in both Devpost and the README.
5. Confirm `outputs/detector.joblib` (or equivalent complete inference bundle) is committed in the
   repo, not gitignored, per the competition's open-source requirement. Do not commit regenerated
   feature caches or datasets.
6. Run through the global definition-of-done checklist in [../README.md](../README.md).
7. Push the final commit, verify the repo is public, and submit via Devpost.

## Definition of done

- [ ] `outputs/trade_offs.md` written and referenced from the README.
- [ ] `outputs/ablation_table.csv` and `outputs/robustness_table.csv` are linked and accurately
   described without presenting reduced-protocol results as a full sweep.
- [ ] README is complete and accurate to the final code state, including the non-replication note.
- [ ] LICENSE (MIT/Apache) confirmed present.
- [ ] Complete CPU-loadable model artifact confirmed committed/published in the repo.
- [ ] Devpost description is drafted and submitted.
- [ ] Demo video is recorded, public on YouTube, and linked.
- [ ] All 5 required deliverables (§5.5) are present and cross-linked.
- [ ] Final repo state is pushed; `predict.py --device cpu` is run in a fresh process to confirm
   packaging did not break inference.

## Time budget

2 hours 30 minutes for documentation, verification, video, and submission, followed by a separate
1-hour emergency buffer. Submit by hour 11 rather than beginning submission at the deadline.
