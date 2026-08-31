# Phase 1 — Data Acquisition

Goal: make a licensed binary train/test dataset available without blocking implementation, record a
deterministic manifest, and attempt the exact validation-only WildFake subset only after required
evaluation artifacts exist.

Reference: [../../problem_statement.md §5.4](../../problem_statement.md#54-available-resources--data)

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

- [ ] CIFAKE is populated, or a documented fixed SID_Set manifest is available.
- [ ] A balanced 200-image development manifest and the final seeded train/test manifests exist.
- [ ] Image counts, class counts, source URLs, licenses, split policy, and seed are recorded.
- [ ] Every sampled image decodes and labels map consistently to authentic/AIGC.
- [ ] If WildFake is present, path/hash overlap checks prove it is absent from training.

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
