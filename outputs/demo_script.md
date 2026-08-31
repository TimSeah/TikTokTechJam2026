# Demo Script (Target: 3 Minutes 20 Seconds)

## Before Recording

- Use a fresh terminal at the repository root, with dependencies installed and OpenCLIP weights already cached. Keep `data/smoke-input` and the PDF open in advance.
- Record at 1080p with readable terminal zoom. Hide usernames, local secrets, notifications, and unrelated browser tabs.
- Use only project-owned, generated, public-domain, or appropriately licensed visuals and audio. Avoid third-party logos and copyrighted music unless permission is documented.

## 0:00-0:25 - Problem and Impact

**On screen:** README title, then the first page of `docs/technical_report.pdf`.

**Say:**

> Social platforms resize, recompress, blur, crop, and recolour images. An AI-image detector must
> survive those changes without pretending that one benchmark proves universal detection. This
> project produces an image-level fake-confidence score and evaluates both transformation robustness
> and transfer to different source pipelines.

## 0:25-0:50 - Final Design and Insight

**On screen:** Figure 1 in the report, then point to the semantic path.

**Say:**

> The promoted detector uses a frozen 151-million-parameter OpenCLIP image encoder and a small
> standardized logistic-regression head, comfortably below the two-billion-parameter limit. I first
> tested a semantic-plus-Fourier hybrid. Its failure under domain shift became the key insight: a
> plausible forensic feature can become a shortcut when its scale leaves the training distribution.

## 0:50-1:30 - Live End-to-End Inference

**On screen:** Run from the repository root:

```powershell
python src/predict.py --input_dir data/smoke-input --out outputs/preds.json --device cpu
Get-Content outputs/preds.json
```

**Say while the output is visible:**

> The inference script recursively accepts an image directory and writes one deterministic JSON
> record per image. `image_path` identifies the input and `pred` is the continuous fake-confidence
> score. The same model artifact loads on CPU or GPU; the score supports ranking and thresholding,
> but it is not a deployment-calibrated probability.

Pause long enough for both required JSON fields to be readable.

## 1:30-2:05 - Clean and Transformed Results

**On screen:** Figure 2 and the summary rows in `outputs/robustness_table.csv`.

**Say:**

> On 20,000 held-out CIFAKE images, the promoted model reaches 0.9578 clean ROC AUC and 0.8966 mean
> AUC across all 14 required JPEG, blur, resize, noise, colour-jitter, and crop conditions. The
> clean-and-robust composite is 0.9272. The hardest cases are quarter-scale resizing, heavy blur, and
> high noise. On the held-out native-domain promotion gates, AUC is 0.9919 on SID, 0.9029 on
> COCO/DALL-E, and 0.9127 on LAION/DALL-E.

Do not describe the promotion gates as a final independent benchmark; they influenced model
selection.

## 2:05-2:35 - Failure, Diagnosis, and Iteration

**On screen:** Figure 4, then Table 5 in the report.

**Say:**

> The original hybrid scored 0.9723 on the CIFAKE composite, yet its blind probabilities collapsed:
> almost every score rounded to exactly one, so every image was classified fake. Raw margins still
> retained AUC 0.7598 on SID and 0.7140 on LAION/DALL-E, but only 0.5377 on COCO/DALL-E. Frequency
> values reached about 42 times the clean-CIFAKE maximum. I removed that branch, broadened training,
> and added saturation and out-of-distribution feature checks to promotion.

## 2:35-3:00 - Errors and Trade-Offs

**On screen:** `outputs/error_analysis.md`; show `false_positive_1.jpg`, then
`false_negative_1.jpg`.

**Say:**

> Heavy blur can erase natural texture, producing false positives like this real image scored 0.996.
> It can also hide synthesis artifacts, producing false negatives like this generated image scored
> 0.009. The final model trades some CIFAKE performance for better multi-domain behavior. It should
> support human review, never act as sole evidence for moderation, authorship, or fraud decisions.

## 3:00-3:20 - Reproducibility and Close

**On screen:** repository tree, `README.md`, and the report's claim-to-artifact map.

**Say:**

> The public repository includes setup, the required inference command, deterministic manifests,
> model hashes, metrics, error examples, limitations, and report rebuild steps. The strongest claim is
> measured robustness to specified transformations plus held-out validation under compound domain
> shift, not universal AI-image detection. The next test must use sources and generators that did not
> influence fitting, calibration, or promotion.

## Submission Checklist

- Confirm the public repository opens without authentication and the README contains overview, setup, reproduction, limitations, improvements, and team contributions where applicable.
- Confirm the video visibly demonstrates directory input and JSON output containing `image_path` and `pred`, plus clean/transformed results, a false positive, a false negative, and trade-offs.
- Upload the final 2-4 minute video to YouTube with public visibility; test the URL in a signed-out browser window.
- Add the same YouTube URL to both the README and Devpost submission.
- Recheck every displayed image, logo, font, and audio track for licensing or permission.
