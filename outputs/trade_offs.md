# Trade-Offs

## Robustness vs. Clean Accuracy

The original CIFAKE-only experiment showed that augmentation was the main transformation-robustness
contribution. Adding transformed views to the clean hybrid lowered clean AUC from `0.991760` to
`0.988409`, but raised condition-weighted robust AUC from `0.910941` to `0.956211` and the composite
score from `0.951351` to `0.972310`.

The promoted `semantic_native_mixed` model makes a different trade-off. It reaches `0.957820` clean
AUC, `0.896636` robust AUC, and a `0.927228` composite on CIFAKE. Those values are lower than the
original hybrid's because the model removes the brittle FFT branch and fits a balanced,
multi-domain sample instead of specializing on all 100,000 CIFAKE training images. In return, it
avoids the hybrid's native-resolution score saturation and passes every cross-domain gate.

## Generalization vs. Specialization

The 14-condition CIFAKE sweep measures transformation robustness, not transfer to unseen sources.
Frozen diagnostics made that distinction concrete: the original hybrid scored `0.4975`, `0.5025`,
and `0.5000` AUC on SID and two WildFake samples, predicting every image as fake. Its semantic
component retained useful ranking, which implicated the frequency branch as a low-level CIFAKE
shortcut.

The promoted model fits evaluation-disjoint CIFAKE, SID-Set, and WildFake partitions and requires
cross-domain gates before promotion. It reaches `0.991900` SID validation AUC and `0.902925` /
`0.912700` on the two WildFake evaluations. This is much stronger evidence than CIFAKE-only model
selection, but each native gate has only 400 images and the WildFake fitting data covers nine source
groups. The project therefore claims measured multi-domain improvement, not universal AI-image
detection.

## Complexity vs. Feasibility

Freezing CLIP and fitting logistic regression makes training cheap, deterministic, CPU-loadable,
and easy to ablate. The cost is that the semantic representation cannot adapt and inference still
loads a 151M-parameter backbone. The promoted head has only 512 coefficients and one intercept, and
the complete semantic cache, fit, gate, and promotion pipeline ran in 547.063 seconds on the RX
7900 XTX.

Removing the fixed FFT branch reduces feature complexity and eliminates its extreme out-of-domain
logit contribution, but also gives up some in-distribution discrimination. Fine-tuning CLIP or
reintroducing a bounded frequency branch might improve quarter-scale resize, heavy blur, and heavy
noise performance, but each option would add compute or shortcut risk and would need to pass the
same disjoint promotion gates.
