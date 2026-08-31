# Phase 6 — Deliverables Packaging: Completion Record

Original goal: assemble and submit all required Track 5 deliverables.

Reference: [../../problem_statement.md §5.5](../../problem_statement.md#55-expected-deliverables)

## Steps

1. Write `outputs/trade_offs.md`: a short, explicit discussion of robustness vs. clean accuracy,
   generalization vs. specialization, and complexity vs. feasibility — the three trade-offs the
   workshop slides ask judges to look for (Slide 10).
2. Write the top-level `README.md`: project overview, setup/install instructions, reproduction
   steps for native AMD, Colab/CUDA, and CPU inference (exact commands per phase), limitations
   (be explicit — e.g. CIFAKE is low-resolution
   32×32, reduced severity/augmentation sweep if scope was cut), an explicit **"why this isn't a
   direct replication"** note, the current semantic multi-domain training and promotion protocol,
   the historical hybrid ablation and failure evidence, confirmation that the LICENSE (MIT/Apache)
   is present, dataset attribution, and team contributions.
3. Write the Devpost "Written Project Description": problem framing, approach (hybrid CLIP +
   frequency-domain detector trained with augmentation), tools/APIs/libraries/datasets used, and the
   headline Final Score + cross-generator AUC from Phase 3. Before external publication, revise
   this original draft to make the promoted semantic model and its completed gates the current
   result.
4. Record a 2–4 minute demo video: show CPU-capable `predict.py` running on a folder, inspect the
   output JSON, then walk through the headline robustness and ablation tables plus 1–2 error
   examples. Show cross-generator results only if the exact validation subset was used. Upload to
   YouTube as public, and link it in both Devpost and the README.
5. Confirm `outputs/detector.joblib` (or equivalent complete inference bundle) is committed in the
   repo, not gitignored, per the competition's open-source requirement. Do not commit regenerated
   feature caches or datasets.
6. Run through the global definition-of-done checklist in [../README.md](../README.md).
7. Push the final commit, verify the repo is public, and submit via Devpost.

## Outcome

The repository contains the CPU-loadable model, inference CLI, source tables, model card, error
analysis, trade-off report, Devpost copy, demo script, and a deployed interactive React + FastAPI
experience. The live frontend is <https://real-vs-ai.pages.dev>; inference is served by Modal. This
documentation pass adds browser screenshots, a full report of the original hybrid and its failure,
and tracked cross-domain evidence. The final retraining pass adds the promoted
`semantic_native_mixed` artifact, current model card and CIFAKE tables, `native_metrics.json`, and
`native_training_timing.json`. The existing technical report and Devpost draft predate this final
promotion and must be refreshed before they are published as current-model documents. Public video
publication and final Devpost submission remain external actions and are intentionally not marked
complete.

Chrome DevTools verification after retraining confirmed that the hosted API is healthy but still
reports `ViT-B-32-quickgelu / hybrid_augmented`. The checked-in and local-demo default is
`semantic_native_mixed`; redeploying Modal and replacing the historical screenshots are therefore
remaining release steps rather than completed-model evidence.

## Definition of done

- [x] `outputs/trade_offs.md` is written and referenced from the README.
- [x] `outputs/ablation_table.csv` and `outputs/robustness_table.csv` are linked and accurately
   described without presenting reduced-protocol results as a full sweep.
- [x] README is accurate to the implemented detector and includes the non-replication note.
- [x] LICENSE (MIT) is present.
- [x] The complete CPU-loadable `outputs/model.joblib` artifact is included in the repository.
- [ ] The original Devpost draft in `outputs/devpost_description.md` is revised for the promoted
   semantic model before publication.
- [x] The robustness summary, error analysis, inference CLI, repository documentation, and demo
      application are present and cross-linked.
- [x] Final promotion evidence is recorded in `outputs/native_metrics.json` and
   `outputs/native_training_timing.json`; every gate and training stage passed.
- [x] `predict.py --device cpu` passed in a fresh process with the packaged model.
- [ ] Modal is redeployed with `semantic_native_mixed` and the README screenshots are recaptured.
- [ ] Demo video is recorded, public on YouTube, and linked.
- [ ] Final Devpost submission and repository publication state are verified.

## Time budget

2 hours 30 minutes for documentation, verification, video, and submission, followed by a separate
1-hour emergency buffer. Submit by hour 11 rather than beginning submission at the deadline.
