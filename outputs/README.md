# Outputs

Versioned submission artifacts live here:

- `model.joblib`: promoted CPU-loadable `semantic_native_mixed` detector bundle;
- `model-native.joblib`, `native_metrics.json`, and `native_training_timing.json`: gated candidate,
  SID/WildFake promotion metrics, and evidence that training completed within 60 minutes;
- `robustness_summary.json`, `robustness_table.csv`, and `ablation_table.csv`: current CIFAKE
  clean/transformed evaluation evidence;
- `cross_domain_summary.json`: frozen-model SID/WildFake diagnostics, component ablations, and
  the original hybrid failure analysis;
- `model_card.md`, `error_analysis.md`, and `trade_offs.md`: scope, failures, and responsible-use
  notes; and
- `devpost_description.md` and `demo_script.md`: submission copy and recording plan.

Feature caches, datasets, and regenerated `preds.json` are intentionally not committed.
