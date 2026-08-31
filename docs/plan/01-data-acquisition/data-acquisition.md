# Phase 1 — Data Acquisition

Goal: download and organize training data (CIFAKE) and the validation-only subset (COCO val2017 vs.
DALL·E Advanced from WildFake), with a directory structure ready for Phases 2–3.

Reference: [../../problem_statement.md §5.4](../../problem_statement.md#54-available-resources--data)

## Steps

1. Download CIFAKE from Kaggle
   (`birdy654/cifake-real-and-ai-generated-synthetic-images`) — requires a Kaggle account/API
   token. If blocked, fall back to SID_Set from Hugging Face (no auth required).
2. Unzip into `data/cifake/{train,test}/{REAL,FAKE}/`.
3. Obtain the validation-only subset: WildFake (translate the ModelScope page first) — extract
   specifically the COCO val2017 (4,998 non-AIGC) and DALL·E Advanced (8,843 AIGC) images. If that
   exact split is hard to isolate quickly, substitute a demo-only subset of comparable structure and
   clearly label it in the README as **not** the official validation subset.
4. Write `data/README.md` noting exactly what is train vs. validation-only, with an explicit
   reminder: never point training scripts at the validation-only folder.
5. Spot-check a handful of images visually to confirm labels look correct.

## Definition of done

- [ ] `data/cifake/train/REAL`, `data/cifake/train/FAKE`, `data/cifake/test/REAL`,
      `data/cifake/test/FAKE` are populated.
- [ ] `data/validation_only/` is populated with the COCO/DALL·E demo subset (or a documented
      substitute), clearly separated from training data.
- [ ] Image counts are logged (expected: CIFAKE ~100k total; validation-only ~13.8k).
- [ ] No validation-only images are referenced anywhere in Phase 2 training code.

## Time budget

30 minutes.

## Risks

- Kaggle auth friction → use SID_Set (Hugging Face) as the primary training set instead.
- WildFake ModelScope access/translation friction → substitute a smaller public real/fake sample
  set and document it clearly as a stand-in.
