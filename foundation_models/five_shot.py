"""Five-shot target-cohort adaptation.

Starting point: the best-validation Pool A full fine-tuning checkpoint written by
train_pool.py --mode fft (backbone and classification head). For each seed in 42-46:
  * five labeled windows per class are sampled from the pooled Pool B training split
    (sample_support_indices, identical for all model families),
  * only the classification head is updated, with the backbone frozen and in eval mode,
    for a fixed budget of 30 epochs (AdamW, constant lr 5e-4, betas (0.9, 0.95),
    weight decay 0.01, eps 1e-7), without early stopping or target-cohort validation,
  * the final-epoch model is evaluated on the fixed Pool B test split.

Example:
  python five_shot.py --model chronos2 --model_size base --data_dir /path/to/NinaPro_DB1 \
      --dataset_name NinaPro_DB1 --pool_a_ckpt results/chronos2_base_fft/NinaPro_DB1_fft_best.pt \
      --out_dir results/chronos2_base_five_shot
"""
import argparse
import json
import statistics
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

import models
from common import data, protocol


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(models.FAMILIES), required=True)
    p.add_argument("--model_size", choices=models.ALL_SIZES, default="base")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--pool_a_ckpt", required=True, help="best FFT checkpoint from train_pool.py")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--batch_size", type=int, default=None)
    args = p.parse_args()

    family = models.get_family(args.model, args.model_size)
    name = models.display_name(family, args.model_size)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = data.load_pool_data(args.data_dir)
    b_train, b_test = d["splits"]["b_train"], d["splits"]["b_test"]
    if b_train is None or b_test is None:
        raise ValueError("Pool B train/test splits are required for five-shot adaptation")
    bs = args.batch_size or family.batch_size(args.model_size, args.dataset_name, d["n_channels"])
    query_loader = DataLoader(b_test, batch_size=bs, shuffle=False)
    state = torch.load(args.pool_a_ckpt, map_location="cpu", weights_only=True)

    rows = []
    for seed in protocol.FEWSHOT_SEEDS:
        idx = protocol.sample_support_indices(b_train.labels, d["n_classes"], protocol.FEWSHOT_N_PER_CLASS, seed)
        support_loader = DataLoader(Subset(b_train, idx), batch_size=min(bs, len(idx)), shuffle=True)
        model = family.build_classifier(args.model_size, d["n_channels"], d["n_classes"])
        model.load_state_dict(state)
        model.to(protocol.DEVICE)
        protocol.adapt_head(model, support_loader, seed)
        metrics = protocol.evaluate(model, query_loader)
        print(f"seed={seed} n_support={len(idx)} acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f}")
        rows.append({"seed": seed, "n_support": int(len(idx)), "n_test": len(b_test), **metrics})
        del model
        protocol.free_gpu_memory()

    f1 = [r["f1"] * 100 for r in rows]
    summary = {"dataset": args.dataset_name, "model": name,
               "macro_f1_mean": statistics.mean(f1), "macro_f1_std": statistics.stdev(f1), "per_seed": rows}
    print(f"{args.dataset_name}: macro-F1 {summary['macro_f1_mean']:.1f} +- {summary['macro_f1_std']:.1f}")
    (out_dir / f"{args.dataset_name}_five_shot.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
