import argparse
import json

import torch

from src.config import Config
from src.eval import metrics_table
from src.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Ovarian cancer classification pipeline (paper reproduction)")
    parser.add_argument("--data", default="Dataset", help="path to dataset root (Ovarian_Cancer/ Ovarian_Non_Cancer)")
    parser.add_argument("--out", default="output", help="output directory")
    parser.add_argument("--arch", default="attresnet50", choices=["attresnet50"])
    parser.add_argument("--filter", default="bilateral", choices=["bilateral", "clahe", "histeq", "gaussian"])
    parser.add_argument("--fs", default="shap", choices=["shap", "pca", "lasso", "mrmr"])
    parser.add_argument("--n-selected", type=int, default=500)
    parser.add_argument("--fe-epochs", type=int, default=30)
    parser.add_argument("--cls-epochs", type=int, default=200)
    parser.add_argument("--fe-batch", type=int, default=16)
    parser.add_argument("--cls-batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="cpu, cuda, mps, or auto")
    parser.add_argument("--fe-lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--no-augment", action="store_true", help="disable data augmentation")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = Config(
        data_dir=args.data,
        output_dir=args.out,
        seed=args.seed,
        filter_mode=args.filter,
        fe_arch=args.arch,
        fe_epochs=args.fe_epochs,
        fe_batch=args.fe_batch,
        fe_lr=args.fe_lr,
        fe_weight_decay=args.weight_decay,
        fe_momentum=args.momentum,
        cls_epochs=args.cls_epochs,
        cls_batch=args.cls_batch,
        fs_method=args.fs,
        n_selected=args.n_selected,
        augment=not args.no_augment,
    )
    cfg.save(f"{args.out}_config.json")

    results = run_pipeline(cfg, device=device, root=args.data, outdir=args.out)
    table = metrics_table(results)
    print("\n===== Final Results (test set) =====")
    print(table.round(4).to_string())
    table.round(4).to_csv(f"{args.out}/results.csv")
    with open(f"{args.out}/results.json", "w") as f:
        json.dump({k: {kk: (float(vv) if isinstance(vv, (int, float)) else vv) for kk, vv in v.items()} for k, v in results.items()}, f, indent=2)


if __name__ == "__main__":
    main()