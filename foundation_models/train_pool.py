"""Linear probing (LP), full fine-tuning (FFT) and train-from-scratch on Pool A.

The model is trained on Pool A (approximately 90% of subjects) and evaluated on
  * the Pool A test split (seen users; in-distribution tables), and
  * the Pool B test split without any adaptation (zero-calibration held-out users).
FFT and scratch use the same architecture, data, batch size and recipe and differ only in
backbone initialization. The best-validation FFT checkpoint is the starting point for five-shot.

Example:
  python train_pool.py --model chronos2 --model_size base --mode fft \
      --data_dir /path/to/NinaPro_DB1 --dataset_name NinaPro_DB1 --out_dir results/chronos2_base_fft
"""
import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

import models
from common import data, protocol


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(models.FAMILIES), required=True)
    p.add_argument("--model_size", choices=models.ALL_SIZES, default="base")
    p.add_argument("--mode", choices=["lp", "fft", "scratch"], required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--dataset_name", required=True, help="short dataset name, also used for batch-size lookup")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--batch_size", type=int, default=None, help="override the default batch size")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    protocol.set_seed(args.seed)
    family = models.get_family(args.model, args.model_size)
    name = models.display_name(family, args.model_size)
    if not getattr(family, "PRETRAINED", True) and args.mode != "scratch":
        raise ValueError(f"{name} has no pretrained checkpoint; only --mode scratch applies")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = data.load_pool_data(args.data_dir)
    s = d["splits"]
    print(f"{args.dataset_name}: Pool A {d['n_pool_a']} subjects, Pool B {d['n_pool_b']} subjects, "
          f"channels={d['n_channels']}, classes={d['n_classes']}, seq_len={d['seq_len']}")

    bs = args.batch_size or family.batch_size(args.model_size, args.dataset_name, d["n_channels"])
    loader = lambda ds, shuffle: DataLoader(ds, batch_size=bs, shuffle=shuffle,
                                            num_workers=args.num_workers, pin_memory=True)
    init = "random" if args.mode == "scratch" else "pretrained"
    model = family.build_classifier(args.model_size, d["n_channels"], d["n_classes"], init).to(protocol.DEVICE)
    print(f"{name} mode={args.mode} lr={protocol.LR[args.mode]} batch_size={bs}")

    ckpt = out_dir / f"{args.dataset_name}_{args.mode}_best.pt"
    best_val_f1 = protocol.train_with_early_stopping(model, loader(s["a_train"], True), loader(s["a_val"], False),
                                                     args.mode, ckpt)
    result = {
        "dataset": args.dataset_name, "model": name, "mode": args.mode, "lr": protocol.LR[args.mode],
        "batch_size": bs, "best_val_f1": best_val_f1,
        "n_channels": d["n_channels"], "n_classes": d["n_classes"],
        "pool_a_test": protocol.evaluate(model, loader(s["a_test"], False)),
    }
    if s["b_test"] is not None:
        result["pool_b_test_zero_calibration"] = protocol.evaluate(model, loader(s["b_test"], False))
    print(json.dumps(result, indent=2))
    (out_dir / f"{args.dataset_name}_{args.mode}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
