import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def fmt(x):
    return f"{x * 100:.2f}%" if isinstance(x, (int, float)) else str(x)


def main():
    results = pd.read_csv(ROOT / "output" / "results.csv", index_col=0)
    cfg = json.loads((ROOT / "output_config.json").read_text())

    paper = {
        "AttCNN": {
            "ACC": 0.9909,
            "AUC": 0.9985,
            "PRE": 0.9917,
            "SN": 0.9896,
            "SP": 0.9921,
            "F1": 0.9906,
            "MCC": 0.9818,
        }
    }

    full_names = {
        "DT": "Decision Tree",
        "RF": "Random Forest",
        "XGBoost": "eXtreme Gradient Boosting",
        "AdaBoost": "Adaptive Boosting",
        "AttCNN": "Attention-based Convolutional Neural Network",
        "AttResNet50": "Attention-augmented Residual Network-50 (feature extractor)",
    }

    metrics = ["ACC", "SN", "SP", "PRE", "F1", "MCC", "AUC"]
    meta = {
        "ACC": "Accuracy",
        "SN": "Sensitivity / Recall",
        "SP": "Specificity",
        "PRE": "Precision",
        "F1": "F1-score",
        "MCC": "Matthews Correlation Coefficient",
        "AUC": "Area under ROC curve",
    }

    lines = []
    w = lines.append
    w("=" * 100)
    w("OVARIAN CANCER CLASSIFICATION - DETAILED PAPER vs REPRODUCTION COMPARISON")
    w("=" * 100)
    w("")
    w("Paper: Hosen et al., 'Ovarian cancer detection and classification using")
    w("attention-based CNN and deep feature extraction techniques', Scientific")
    w("Reports (2026), DOI: 10.1038/s41598-026-67832-z")
    w("Dataset: STRAMPN - 987 histopathological images (481 OC / 506 Non-OC)")
    w("")

    w("--------------------------------------------------------------------")
    w("1. MODELS AND THEIR FULL FORMS")
    w("--------------------------------------------------------------------")
    w("")
    w(f"{'Short name':<12} {'Role':<30} Full form")
    w(f"{'-'*10:<12} {'-'*28:<30} {'-'*60}")
    for name, ff in full_names.items():
        role = "Feature extractor" if "feature extractor" in ff else "Classifier"
        w(f"{name:<12} {role:<30} {ff}")
    w("")

    w("--------------------------------------------------------------------")
    w("2. HYPERPARAMETERS USED IN THIS REPRODUCTION (output_config.json)")
    w("--------------------------------------------------------------------")
    w("")
    hype = [
        ("Split", f"{cfg['train_ratio']:.0%} / {cfg['val_ratio']:.0%} / {cfg['test_ratio']:.0%} (train/val/test)"),
        ("Image size", f"{cfg['img_size']}"),
        ("Preprocessing filter", cfg["filter_mode"]),
        ("Data augmentation", str(cfg["augment"])),
        ("Feature extractor", f"{cfg['fe_arch']} (epochs={cfg['fe_epochs']}, batch={cfg['fe_batch']}, lr={cfg['fe_lr']})"),
        ("Features extracted", str(cfg["n_features"])),
        ("Feature selection", f"{cfg['fs_method']} (top {cfg['n_selected']} features)"),
        ("Classifier epochs", f"{cfg['cls_epochs']} (batch={cfg['cls_batch']}, lr={cfg['cls_lr']})"),
    ]
    for k, v in hype:
        w(f"  {k:<22}: {v}")
    w("")
    w("  Note: the paper reports split sizes 790/169/169 (approx 80/17/17); this")
    w("  reproduction uses the exact 70/15/15 ratio on 987 images = 690/148/149.")
    w("")

    w("--------------------------------------------------------------------")
    w("3. METRIC LEGEND")
    w("--------------------------------------------------------------------")
    w("")
    for k, v in meta.items():
        w(f"  {k:<4} = {v}")
    w("")

    w("--------------------------------------------------------------------")
    w("4. PER-MODEL COMPARISON  [Paper proposed (AttCNN) vs Reproduction]")
    w("--------------------------------------------------------------------")
    w("")
    header = f"{'Metric':<8}{'Paper model':>14}{'Repro AttCNN':>14}{'Repro best':>14}{'Delta (Repro-Proposed)':>24}"
    w(header)
    w("-" * (len(header)))
    for m in metrics:
        paper_v = paper["AttCNN"][m]
        my_v = results.loc["AttCNN", m]
        best_v = results[m].max()
        delta = (my_v - paper_v) * 100
        w(
            f"{m:<8}{fmt(paper_v):>14}{fmt(my_v):>14}{fmt(best_v):>14}{delta:>+21.2f} pp"
        )
    w("")
    w("  Note: 'Repro best' is the best value across DT/RF/XGBoost/AdaBoost/AttCNN")
    w("  in this reproduction run.")
    w("")

    w("--------------------------------------------------------------------")
    w("5. REPRODUCTION RESULTS - ALL CLASSIFIERS (test set)")
    w("--------------------------------------------------------------------")
    w("")
    w(f"{'Model':<10}{''.join(f'{m:>9}' for m in metrics)}{'time(s)':>9}")
    w("-" * 88)
    for name, row in results.iterrows():
        w(f"{name:<10}{''.join(f'{row[m]*100:>8.2f}%' for m in metrics)}{row['time(s)']:>9.2f}")
    w("")

    w("--------------------------------------------------------------------")
    w("6. DATA SPLIT USED (this reproduction)")
    w("--------------------------------------------------------------------")
    w("")
    w("  Total images : 987 (481 Ovarian_Cancer / 506 Ovarian_Non_Cancer)")
    w("  train        : 690 (70%)")
    w("  val          : 148 (15%)")
    w("  test         : 149 (15%)  - stratified holdout")
    w("")

    w("--------------------------------------------------------------------")
    w("7. ANALYSIS / KEY OBSERVATIONS")
    w("--------------------------------------------------------------------")
    w("")
    for line in [
        "  1. The reproduction's AttCNN (ACC 98.66%) is within ~0.4pp of the paper's",
        "     proposed model (ACC 99.09%, AUC 0.9985, MCC 0.9818).",
        "  2. Accuracy ties at 98.66% for RF, XGBoost, AdaBoost and AttCNN; XGBoost has",
        "     the best AUC (0.9989) in this reproduction.",
        "  3. Specificity and precision reach 100% for all strong classifiers (no false",
        "     positives on the test set); sensitivity is 97.37% (one OC image missed).",
        "  4. DT is clearly the weakest classifier (ACC 94.63%, MCC 0.8938, AUC 0.9457).",
        "  5. Likely gaps vs. the paper: split ratio (70:15:15 vs. ~80:17:17), number of",
        "     training epochs (FE=30, CLS=200) and augmentation/filter settings.",
        "  6. AttCNN time(s)=0.0 in the CSV is a placeholder (see pipeline.py) and does",
        "     not reflect real training time.",
    ]:
        w(line)
    w("")
    w("=" * 100)
    w("Paper values decoded from the reference PDF tables and cross-checked with the")
    w("publisher abstract (accuracy 99.09%, precision 99.17%).")
    w("=" * 100)

    out = ROOT / "comparison_report.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()