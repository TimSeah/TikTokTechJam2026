from src.detector.native_stress import condition_kind, summarize_conditions


def test_condition_kind_and_summary_separate_chains() -> None:
    results = {
        "clean": {"kind": "clean", "probability_ranking": {"auc": 0.9}},
        "jpeg_q70": {
            "kind": "individual_transform",
            "probability_ranking": {"auc": 0.8},
        },
        "blur_sigma1.0": {
            "kind": "individual_transform",
            "probability_ranking": {"auc": 0.6},
        },
        "chain_moderate_reupload": {
            "kind": "platform_style_chain",
            "probability_ranking": {"auc": 0.5},
        },
    }

    summary = summarize_conditions(results)

    assert condition_kind("clean") == "clean"
    assert condition_kind("chain_moderate_reupload") == "platform_style_chain"
    assert condition_kind("jpeg_q70") == "individual_transform"
    assert summary["individual_transform"]["mean_auc"] == 0.7
    assert summary["individual_transform"]["composite_with_clean"] == 0.8
    assert summary["platform_style_chain"]["mean_auc"] == 0.5
