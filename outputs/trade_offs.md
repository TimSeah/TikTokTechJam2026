# Trade-Offs

## Robustness vs. Clean Accuracy

The clean hybrid has the best clean AUC (0.991760), but its condition-weighted robust AUC is only
0.910941. Adding transformed copies lowers clean AUC by 0.003351 while raising robust AUC by
0.045270 and the composite score by 0.020959. The final model therefore accepts a small clean-data
cost for a much larger robustness gain.

## Generalization vs. Specialization

The detector is specialized to CIFAKE's 32x32 real and generated distributions. CLIP supplies a
broad semantic prior, but the FFT branch can learn low-level dataset cues. The 14-condition sweep
tests transform robustness on held-out images, not unseen generators. The exact WildFake reference
set was not acquired within its optional time gate, so no cross-generator claim is made.

## Complexity vs. Feasibility

Freezing CLIP and fitting logistic regression makes training cheap, deterministic, CPU-loadable,
and easy to ablate. The cost is that the semantic representation cannot adapt and inference still
loads a 151M-parameter backbone. A learned frequency encoder or end-to-end fine-tuning might improve
worst-case performance, but would increase overfitting risk, compute, and packaging complexity.

The fixed FFT branch is nearly free and improves clean AUC, yet its unaugmented version is especially
fragile to blur and resizing. Augmentation is what makes the hybrid useful under those conditions;
the branch alone is not a robustness guarantee.