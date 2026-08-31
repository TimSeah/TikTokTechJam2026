# Phase 1 — Data Acquisition: Completion Record

Original goal: make a licensed binary train/test dataset available without blocking implementation, record a
deterministic manifest, and attempt the exact validation-only WildFake subset only after required
evaluation artifacts exist.

Reference: [../../problem_statement.md §5.4](../../problem_statement.md#54-available-resources--data)

## Outcome

CIFAKE was acquired from Kaggle and its published split was preserved under `data/downloads/`.
Seed `2026` produced a 100,000-image training manifest, a 20,000-image held-out test manifest, and a
balanced 200-image development manifest in `data/manifests/`. The source, license, class mapping,
counts, and local layout are recorded in [data/README.md](../../../data/README.md).

The official WildFake manifest counts were inspected, but the exact DALL-E Advanced source required
a 26 GB archive and exceeded the optional acquisition gate. It was not used for training or
full-scale evaluation, and no smaller result is presented as the full organizer reference. After
the submitted artifact was frozen, three versioned, balanced 400-image samples were used for
cross-domain diagnosis.

Post-submission remediation then acquired evaluation-disjoint SID-Set and WildFake-Sample training
partitions. The promoted model uses 4,000 REAL and 4,000 FAKE source images from each of CIFAKE,
SID-Set, and WildFake, for 24,000 source images and 48,000 clean-plus-augmented fitting rows. A
separate 500-per-class SID split was used only to calibrate the decision threshold. Frozen SID and
WildFake evaluation IDs and content hashes were excluded from fitting and calibration. The exact
training revisions, allowed WildFake source groups, exclusions, and promotion results are recorded
in [outputs/model_card.md](../../../outputs/model_card.md) and
[outputs/native_metrics.json](../../../outputs/native_metrics.json).

## Steps

1. Start the CIFAKE download while Phase 0 configures the GPU. Prefer the published Kaggle split and
   stop troubleshooting Kaggle authentication after 30 minutes.
2. Unzip CIFAKE into `data/cifake/{train,test}/{REAL,FAKE}/`. Preserve the published train/test split
   and record image counts and source URL in `data/README.md`.
3. If CIFAKE is still unavailable at the 30-minute gate, stream a fixed, balanced SID_Set sample
   from Hugging Face. Use label 0 (real) and label 1 (full synthetic), exclude label 2 (tampered),
   and do not download the complete 140 GB dataset. Use the published train/validation split rather
   than randomly mixing all rows.
4. Write a seeded manifest containing relative path or dataset row ID, source split, binary label,
   and train/test role. Phase 2 and Phase 3 must consume this manifest rather than independently
   enumerating files.
5. Create a balanced 200-image development manifest immediately so the vertical slice can run while
   the remaining data downloads.
6. Spot-check several images from both labels and verify that every manifest row decodes.
7. After the CIFAKE robustness table exists, spend at most 30 minutes obtaining the exact WildFake
   validation subset: COCO val2017 (4,998 non-AIGC) and DALL·E Advanced (8,843 AIGC). If the exact
   subset cannot be obtained, omit that experiment and document it; do not present a substitute as
   the organizer's WildFake benchmark.
8. Keep WildFake in a separate `data/validation_only/` tree and assert that none of its paths or
   content hashes appear in the training manifest.

## Definition of done

- [x] CIFAKE is populated with the published train/test split.
- [x] A balanced 200-image development manifest and the final seeded train/test manifests exist.
- [x] Image counts, class counts, source URLs, license, split policy, and seed are recorded.
- [x] A 1,000-image validation sample decoded and labels mapped consistently to authentic/AIGC.
- [x] The full WildFake reference was omitted after the acquisition gate; later smaller diagnostic
   samples are clearly separated from it and were not used to fit or calibrate either model.
- [x] Final SID-Set and WildFake fitting partitions contain 4,000 images per class, use recorded
   source revisions, and exclude every frozen evaluation ID and hash.
- [x] The SID calibration split, SID validation split, and both WildFake evaluations remain
   disjoint from the 24,000-image fitting sample.

## Time budget

45 minutes in parallel with environment setup for the primary dataset and development manifest.
WildFake receives a separate 30-minute maximum only after the required robustness evaluation.

## Risks

- Kaggle auth friction → stop at 30 minutes and stream a fixed SID_Set sample rather than downloading
   the full dataset.
- SID_Set volume or streaming latency → cap the balanced sample and persist selected row IDs before
   training so a resumed run uses identical data.
- WildFake ModelScope access/translation friction → omit the optional cross-generator result after
   30 minutes; required CIFAKE evaluation and submission packaging take priority.
