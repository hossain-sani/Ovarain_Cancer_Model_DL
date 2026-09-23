from pathlib import Path

import pandas as pd

PAPER = {
    "ACC": 0.9909,
    "AUC": 0.9985,
    "PRE": 0.9917,
    "SN": 0.9896,
    "SP": 0.9921,
    "F1": 0.9906,
    "MCC": 0.9818,
}

METRICS = ["ACC", "SN", "SP", "PRE", "F1", "MCC", "AUC"]
META = {
    "ACC": "Accuracy",
    "SN": "Sensitivity / Recall",
    "SP": "Specificity",
    "PRE": "Precision",
    "F1": "F1-score",
    "MCC": "Matthews Correlation Coefficient",
    "AUC": "Area under ROC curve",
}
FULL = {
    "EFNet-CBAM": "EfficientNet-B0 + CBAM attention (feature extractor, 1280-d)",
    "DT": "Decision Tree",
    "RF": "Random Forest",
    "XGBoost": "eXtreme Gradient Boosting",
    "AdaBoost": "Adaptive Boosting",
    "AttCNN": "Attention-based Convolutional Neural Network (classifier)",
    "Ensemble": "Soft-voting: RF + XGBoost + AdaBoost + AttCNN",
}


def metrics_table(results):
    return pd.DataFrame(results).T


def fmt(x):
    return f"{x * 100:.2f}%" if isinstance(x, (int, float)) else str(x)


def build_report(outdir, cfg, results, root, baseline_rel="output/results.csv"):
    outdir = Path(outdir)
    baseline_path = Path(root) / baseline_rel
    q = []
    w = q.append
    w("=" * 100)
    w("EXPERIMENT 1 - PAPER vs BASELINE vs EFFICIENTNET-CBAM")
    w("=" * 100)
    w("")
    w("Paper : Hosen et al., Sci. Rep. (2026), 10.1038/s41598-026-67832-z")
    w("        proposed model ACC 99.09% | PRE 99.17% | AUC 0.9985")
    w("")

    w("--------------------------------------------------------------------")
    w("1. MODEL USED")
    w("--------------------------------------------------------------------")
    w("")
    w("  EfficientNet-CBAM (feature extractor)")
    w("    backbone : EfficientNet-B0 (ImageNet pretrained), 1280-d features")
    w("    attention: CBAM = ChannelAttention + SpatialAttention after last block")
    w("    head     : GAP -> Dropout -> Linear(1280, 2)")
    w("  Downstream: SHAP top-500 -> DT / RF / XGBoost / AdaBoost / AttCNN + soft-vote")
    w("")

    w("--------------------------------------------------------------------")
    w("2. CONFIG")
    w("--------------------------------------------------------------------")
    w("")
    for k, v in cfg.__dict__.items():
        w(f"  {k:<22}: {v}")
    w("")

    w("--------------------------------------------------------------------")
    w("3. HEAD-TO-HEAD  (paper vs this experiment, test set)")
    w("--------------------------------------------------------------------")
    w("")
    baseline_best = None
    if baseline_path.exists():
        b = pd.read_csv(baseline_path, index_col=0)
        baseline_best = {m: b[m].max() for m in METRICS}
    header = f"{'Metric':<8}{'Paper':>12}{'Exp AttCNN':>12}{'Exp Ensemble':>14}{'Baseline best':>14}"
    w(header)
    w("-" * len(header))
    for m in METRICS:
        p = fmt(PAPER[m])
        a = fmt(results["AttCNN"][m])
        e = fmt(results["Ensemble"][m])
        bb = fmt(baseline_best[m]) if baseline_best else "n/a"
        w(f"{m:<8}{p:>12}{a:>12}{e:>14}{bb:>14}")
    w("")
    if baseline_best:
        w("  'Baseline best' = best value across DT/RF/XGBoost/AdaBoost/AttCNN of")
        w(f"  the earlier reproduction ({baseline_path}).")
    w("")

    w("--------------------------------------------------------------------")
    w("4. FULL RESULTS - ALL CLASSIFIERS (test set)")
    w("--------------------------------------------------------------------")
    w("")
    w(f"{'Model':<12}{''.join(f'{m:>9}' for m in METRICS)}{'time(s)':>9}")
    w("-" * 90)
    for name, row in results.items():
        w(f"{name:<12}{''.join(f'{row[m]*100:>8.2f}%' for m in METRICS)}{row['time(s)']:>9.2f}")
    w("")

    w("--------------------------------------------------------------------")
    w("5. MODELS - FULL FORMS")
    w("--------------------------------------------------------------------")
    w("")
    for k, v in FULL.items():
        w(f"  {k:<12} {v}")
    w("")

    w("--------------------------------------------------------------------")
    w("6. SUCCESS CRITERIA")
    w("--------------------------------------------------------------------")
    w("")
    checks = [
        ("ACC", 0.9909, results["Ensemble"]["ACC"], ">" ),
        ("AUC", 0.9985, results["Ensemble"]["AUC"], ">="),
        ("MCC", 0.98, results["Ensemble"]["MCC"], ">="),
    ]
    for name, target, val, op in checks:
        ok = val > target if op == ">" else val >= target
        w(f"  Ensemble {name} {val*100:.2f}% vs target {target*100:.2f}% -> {'PASS' if ok else 'FAIL'}")
    w("")
    w("=" * 100)

    (outdir / "comparison_report.txt").write_text("\n".join(q) + "\n", encoding="utf-8")