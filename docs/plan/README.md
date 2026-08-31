# Track 5 Execution Plan — Robust Detection of AI-Generated Images

References:

- [../problem_statement.md](../problem_statement.md#5-robust-detection-of-aigenerated-images-under-realworld-transformations)
- [../problem_statement.md §5.7](../problem_statement.md#57-technical-workshop-notes-supplementary)
  — organizer workshop slides with competition rules, scoring formula, and recommended methodology
  not present elsewhere in the base problem statement.

## Chosen approach

**A hybrid detector: CLIP (ViT-B/32) semantic embeddings fused with a lightweight frequency-domain
feature branch (FFT/DCT-based), trained on augmented CIFAKE data (clean + transformed copies of
each training image) and a shallow classifier head (logistic regression / small MLP).** Evaluated
for robustness against the 6 required transform families, plus a cross-generator generalization
check when the exact WildFake-derived COCO val2017 / DALL·E Advanced validation subset can be
obtained after required evaluation — used **only** for evaluation, never for training.

## Compute strategy and decision gate

The target machine has a Radeon RX 7900 XTX (24 GB VRAM), Ryzen 7 7800X3D, and Windows 11 build
26200. The preferred path is native Windows PyTorch because AMD officially supports the RX 7900 XTX
with PyTorch 2.9.1 + ROCm 7.2.1 on Python 3.12. This avoids installing a new WSL distribution and
avoids slow training-data access through `/mnt/c`.

1. Start a free Google Colab GPU runtime in parallel as a ready fallback; free GPU allocation and
  runtime duration are not guaranteed.
2. Spend at most 45 minutes creating the native Python 3.12 AMD environment and installing the
  official ROCm 7.2.1 wheels. The required Windows graphics driver is Adrenalin 26.2.2.
3. Accept the native path only if `torch.cuda.is_available()` is true, the device name is the RX
  7900 XTX, and a 256-image CLIP embedding benchmark completes successfully.
4. If the native check fails, use the allocated Colab CUDA GPU immediately. Do not debug the native
  stack beyond the 45-minute gate.
5. Set up Ubuntu 22.04/24.04 under WSL2 only if native Windows fails and Colab has no usable GPU or
  terminates. If WSL is needed, keep the repo, dataset, and caches inside the Linux filesystem
  rather than under `/mnt/c`, then copy final artifacts back to the Windows checkout.

All project code must remain device-agnostic (`--device auto`) and the final inference smoke test
must also pass on CPU so reviewers do not need matching GPU hardware.

## Why this approach

- **Avoids disqualification risk.** The workshop slides state explicitly: *"Do not directly
  replicate existing models or approaches"* and *"Just using a pre-trained AI image detection model
  would result in disqualification."* A bare frozen-CLIP-linear-probe is the well-known published
  "Universal Fake Image Detectors" method essentially unmodified — fusing in a frequency-domain
  branch and training on augmented data makes this my own pipeline, not a literal reproduction,
  while still using an explicitly whitelisted pretrained backbone (CLIP is named as approved).
- **Matches the graded methodology, not just the graded metric.** The slides frame "training for
  robustness with augmentation" as the key idea (not merely evaluating robustness after the fact),
  and separately recommend the CLIP-semantics + frequency-patch hybrid as the strongest simple
  design (Slide 7: "Go hybrid"). Building both in directly targets what reviewers were told to look
  for.
- The `<2B` parameter constraint is trivially satisfied (frozen CLIP ViT-B/32 ≈ 151M params + cheap
  frequency features + a linear/shallow head).
- The available 24 GB GPU makes full-dataset batched feature extraction practical. GPU time is used
  to improve coverage, robustness evaluation, and ablation evidence rather than to add a riskier
  trainable backbone.
- It minimizes time spent on model training/tuning and maximizes time spent on the parts that are
  actually judged: the robustness score formula, generalization check, error analysis, and
  reproducible deliverables.

## Timeline (12-hour deadline)

| Time | Phase | Exit condition |
| --- | --- | --- |
| 0:00–0:45 | [Environment](00-environment-setup/environment-setup.md) + [data](01-data-acquisition/data-acquisition.md), in parallel | One GPU path passes the 256-image benchmark; CIFAKE download has started |
| 0:45–2:30 | [Vertical slice](02-baseline-detector/baseline-detector.md) | 200 images run through transform → features → training → saved artifact |
| 2:30–4:00 | [Full feature extraction and training](02-baseline-detector/baseline-detector.md) | Full CIFAKE clean + one-augmentation features and three ablation heads are saved |
| 4:00–4:30 | [Inference CLI](04-inference-script/inference-script.md) | Directory-to-JSON inference passes on GPU and CPU |
| 4:30–6:00 | [Robustness evaluation](03-robustness-pipeline/robustness-pipeline.md) | All six families are measured; all 14 conditions are measured if throughput permits |
| 6:00–6:45 | [Ablation + error analysis](05-error-analysis/error-analysis.md) | Comparison table and at least two FP/two FN examples exist |
| 6:45–8:30 | [Documentation and packaging](06-deliverables-packaging/deliverables-packaging.md) | README, model card, reports, attribution, and Devpost draft are complete |
| 8:30–9:15 | Final verification | Fresh-process CPU smoke test passes; artifacts are committed and pushed |
| 9:15–10:15 | Demo video | A 2–4 minute video is uploaded and publicly accessible |
| 10:15–11:00 | Submission | Devpost is submitted with repository and video links |
| 11:00–12:00 | Emergency buffer | Verify the live submission; fix only submission-blocking defects |

Hard cut lines: stop environment debugging at 0:45, require a working trained artifact by 4:00,
stop expanding evaluation at 6:00, freeze code at 8:30, and submit by 11:00.

## Deliverable mapping (per problem statement §5.5)

| # | Required deliverable | Produced in |
| --- | --- | --- |
| 1 | Written Project Description (Devpost) | Phase 6 |
| 2 | Public Code/GitHub Repository + inference script (image dir → JSON) | Phases 2–4 |
| 3 | Demo Video | Phase 6 |
| 4 | Robustness Evaluation Summary | Phase 3 |
| 5 | Error Analysis Note | Phase 5 |

## Global definition of done (project-level)

- [ ] `python src/predict.py --input_dir <image_dir> --out preds.json` runs end-to-end and produces valid JSON with
      `image_path` + `pred` for every image in the directory.
- [ ] The saved artifact contains feature scaling, classifier, class mapping, backbone/checkpoint,
  preprocessing, frequency-feature configuration, and random seed; it loads on CPU.
- [ ] A robustness table exists comparing clean vs. each of the 6 transform families (at the listed
      severities, or a documented reduced subset) on a held-out test split.
- [ ] The composite score `Final Score = 0.50 × AUC_clean + 0.50 × AUC_robust` is computed and
  reported; if the exact validation-only subset is obtained, cross-generator AUC is also reported.
- [ ] An ablation compares semantic-only clean training, hybrid clean training, and hybrid
  augmented training using the same held-out images.
- [ ] An error-analysis note cites concrete false positive / false negative examples with a stated
      hypothesis for each.
- [ ] A trade-offs write-up covers robustness vs. clean accuracy, generalization vs. specialization,
      and complexity vs. feasibility.
- [ ] README documents setup, reproduction steps, limitations, team contributions, and an explicit
      "why this isn't a direct replication of an existing method" note.
- [ ] No training occurred on the WildFake/COCO–DALL·E validation-only subset (verify before
      submission).
- [ ] Model uses fewer than 2B parameters (confirm and record the count); only publicly available
      pretrained backbones are used (CLIP is explicitly whitelisted).
- [ ] A LICENSE file (MIT or Apache) is present in the repo.
- [ ] Trained model artifact (`outputs/detector.joblib` or equivalent) is committed/published in the
      repo, not gitignored.
- [ ] Repo is pushed, public, and linked in the Devpost description; demo video is uploaded to
      YouTube (public) and linked.

## Target repository layout (for the implementation phase)

```text
<project-root>/
  LICENSE                      # MIT or Apache — required by competition rules
  data/                        # gitignored — CIFAKE + validation-only downloads
    cifake/{train,test}/{REAL,FAKE}/
    validation_only/           # COCO val2017 subset + DALL·E Advanced (demo-only, never trained on)
  src/
    detector/
      transforms.py             # the 6 required augmentation families (built in Phase 2, reused by Phase 3)
      freq_features.py           # lightweight FFT/DCT frequency-domain feature extraction
      embed.py                    # CLIP embedding extraction (clean + augmented copies)
      train_probe.py                # trains the classifier head on fused CLIP + frequency features
      evaluate.py                    # robustness table + Final Score computation
    predict.py                        # required CLI: image dir -> JSON
  outputs/
    detector.joblib                   # complete CPU-loadable inference bundle (committed, not gitignored)
    model_card.md                     # backbone, novelty note, parameter count, accuracy
    robustness_table.csv
    ablation_table.csv
    error_analysis.md
    trade_offs.md
    error_examples/
    preds.json                         # gitignored — regenerated output, not part of the submission artifact set
  README.md
  requirements.txt
```

## Risks & fallbacks

- Native AMD setup exceeds 45 minutes or fails the batch benchmark → switch to the already-open
  Colab GPU; use WSL2 only if Colab is unavailable.
- Colab terminates or loses its GPU → save restartable feature shards and the current model artifact
  after each stage; continue locally or in WSL without repeating completed work.
- CLIP (`open_clip`/OpenAI `clip`) download or execution fails → use torchvision's pretrained
  ResNet18 features while keeping the frequency branch and augmented training.
- CIFAKE download friction exceeds 30 minutes → stream a fixed, balanced SID_Set sample containing
  labels 0 (real) and 1 (full synthetic), excluding label 2 (tampered); do not download all 140 GB.
- WildFake acquisition exceeds 30 minutes → omit the cross-generator experiment and state that the
  organizer's demonstration-only set was not evaluated; do not relabel a substitute as WildFake.
- Time overrun in Phase 2 → keep one random transformed copy per training image and reduce the fixed
  balanced sample size; never remove the frequency branch or the end-to-end inference artifact.
- If the frequency-domain branch turns out to add negligible accuracy → keep it anyway and report
  that finding honestly in the ablation and trade-offs.
- Time overrun in Phase 3 → evaluate one representative severity from every transform family before
  adding the remaining severities, and label any reduced result as a reduced-protocol score.

## Next step

Start [00-environment-setup](00-environment-setup/environment-setup.md) and
[01-data-acquisition](01-data-acquisition/data-acquisition.md) in parallel. The first irreversible
decision is the 45-minute native AMD/Colab compute gate; implementation begins with a 200-image
vertical slice immediately after one GPU path passes.
