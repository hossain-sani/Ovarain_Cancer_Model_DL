import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from pipeline import run_pipeline
from report import build_report, metrics_table


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: EfficientNet-CBAM reproduction boost")
    parser.add_argument("--data", default=str(ROOT / "Dataset"), help="path to dataset root")
    parser.add_argument("--out", default=str(ROOT / "experiments" / "experiment1" / "output"), help="output directory")
    parser.add_argument("--filter", default="bilateral", choices=["bilateral", "clahe", "histeq", "gaussian"])
    parser.add_argument("--fs", default="shap", choices=["shap", "pca", "lasso", "mrmr"])
    parser.add_argument("--n-selected", type=int, default=500)
    parser.add_argument("--fe-epochs", type=int, default=40)
    parser.add_argument("--cls-epochs", type=int, default=200)
    parser.add_argument("--fe-batch", type=int, default=32)
    parser.add_argument("--cls-batch", type=int, default=32)
    parser.add_argument("--fe-lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="cpu, cuda, mps, or auto")
    parser.add_argument("--no-augment", action="store_true", help="disable data augmentation")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        )
    print(f"Device: {device}")

    cfg = Config(
        data_dir=args.data,
        output_dir=args.out,
        seed=args.seed,
        filter_mode=args.filter,
        fe_epochs=args.fe_epochs,
        fe_batch=args.fe_batch,
        fe_lr=args.fe_lr,
        cls_epochs=args.cls_epochs,
        cls_batch=args.cls_batch,
        fs_method=args.fs,
        n_selected=args.n_selected,
        augment=not args.no_augment,
    )
    cfg.save(f"{args.out}_config.json")

    results = run_pipeline(cfg, device=device, root=args.data, outdir=args.out)

    table = metrics_table(results)
    print("\n===== Experiment 1 Final Results (test set) =====")
    print(table.round(4).to_string())
    table.round(4).to_csv(f"{args.out}/results.csv")
    with open(f"{args.out}/results.json", "w") as f:
        json.dump(
            {k: {kk: (float(vv) if isinstance(vv, (int, float)) else vv) for kk, vv in v.items()} for k, v in results.items()},
            f,
            indent=2,
        )

    build_report(args.out, cfg, results, ROOT)
    print(f"\nComparison report -> {args.out}/comparison_report.txt")


if __name__ == "__main__":
    main()