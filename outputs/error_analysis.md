# Error Analysis

The promoted `semantic_native_mixed` model was evaluated on 20,000 held-out CIFAKE images under
Gaussian blur sigma 2.0, its second-weakest condition. At the artifact's calibrated `0.781959`
threshold, the examples below are the
two most confident false positives and false negatives. Scores are `P(FAKE)`; exact source paths,
IDs, and scores are preserved in [error_examples/examples.csv](error_examples/examples.csv).

## False Positives

### 1. Real image predicted fake, score 0.995727

![False positive 1](error_examples/false_positive_1.jpg)

The heavy blur leaves an isolated yellow shape against a nearly black field. Fine photographic
texture and scene context disappear, producing the simple, high-contrast composition of a rendered
icon or synthetic thumbnail.

### 2. Real image predicted fake, score 0.984514

![False positive 2](error_examples/false_positive_2.jpg)

The subject collapses into a bright vertical streak between broad green and blue regions. With
objects and boundaries erased, the semantic embedding has little evidence left to identify a
natural scene.

## False Negatives

### 1. Fake image predicted real, score 0.008672

![False negative 1](error_examples/false_negative_1.jpg)

The blurred generated animal keeps a plausible face-like silhouette and mottled natural colors.
Removing its finer synthesis artifacts leaves a coarse pattern compatible with a low-resolution
wildlife photograph.

### 2. Fake image predicted real, score 0.010321

![False negative 2](error_examples/false_negative_2.jpg)

The generated scene retains several animal-like shapes and neutral photographic tones after local
detail is removed. Its coarse composition remains plausible enough for the semantic-only model to
rank it as real.

## Findings

Quarter-scale resizing causes the largest clean-relative AUC drop, from `0.957820` to `0.774426`.
Heavy blur sigma 2.0 is next at `0.785540`, followed by noise sigma 0.10 at `0.797451`. These
transformations erase or disrupt the image structure available to the frozen semantic embedding.

In the original CIFAKE-only ablation, frequency fusion improved clean AUC from `0.988755` to
`0.991760`, while augmentation raised the hybrid's robust AUC from `0.910941` to `0.956211`.
However, that frequency branch later saturated on native-resolution data and was removed. The
promoted semantic model trades some CIFAKE performance for multi-domain behavior, reaching
`0.896636` robust AUC while passing all SID and WildFake promotion gates. The image-level
explanations above remain hypotheses rather than causal proofs.
