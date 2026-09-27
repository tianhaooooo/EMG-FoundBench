"""Cohort-restricted leave-one-subject-out (LOSO) evaluation, one fold per call.

The cohort is all subjects for datasets with at most 10 subjects, otherwise the first 10 subjects
in canonical subject-ID order. In each fold one cohort subject is the test subject, one of the
remaining cohort subjects is held out for validation-based early stopping, and the others are
used for training. The model starts from the pretrained checkpoint and is trained with the full
fine-tuning recipe (Appendix C.8).

Example:
  python loso.py --model moment --model_size small --data_dir /path/to/loso/NinaPro_DB1 \
      --dataset_name NinaPro_DB1 --test_subject s1 --out_dir results/moment_small_loso
"""
import argparse
import json
import random
from pathlib import Path

from torch.utils.data import DataLoader

import models
from common import data, protocol


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(models.FAMILIES), required=True)
    p.add_argument("--model_size", choices=models.ALL_SIZES, default="base")
    p.add_argument("--data_dir", required=True, help="LOSO dataset folder (<subject>/emg.npy, labels.npy)")
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--test_subject", required=True)
    p.add_argument("--val_subject", default=None, help="default: a fixed seeded choice among the remaining subjects")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    protocol.set_seed(args.seed)
    family = models.get_family(args.model, args.model_size)
    name = models.display_name(family, args.model_size)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort = data.loso_cohort(args.data_dir)
    if args.test_subject not in cohort:
        raise ValueError(f"{args.test_subject} is not in the evaluation cohort {cohort}")
    remaining = [s for s in cohort if s != args.test_subject]
    val_subject = args.val_subject or remaining[random.Random(0).randrange(len(remaining))]
    train_subjects = [s for s in remaining if s != val_subject]
    print(f"cohort={cohort}\ntest={args.test_subject} val={val_subject} train={train_subjects}")

    ds, n_channels, n_classes = data.load_loso_data(args.data_dir, train_subjects, val_subject, args.test_subject)
    bs = args.batch_size or family.batch_size(args.model_size, args.dataset_name, n_channels)
    loader = lambda d, shuffle: DataLoader(d, batch_size=bs, shuffle=shuffle,
                                           num_workers=args.num_workers, pin_memory=True)
    model = family.build_classifier(args.model_size, n_channels, n_classes).to(protocol.DEVICE)
    print(f"{name} LOSO lr={protocol.LR['fft']} batch_size={bs} classes={n_classes}")

    ckpt = out_dir / f"{args.dataset_name}_{args.test_subject}_best.pt"
    best_val_f1 = protocol.train_with_early_stopping(model, loader(ds["train"], True), loader(ds["val"], False),
                                                     "fft", ckpt)
    result = {"dataset": args.dataset_name, "model": name, "test_subject": args.test_subject,
              "val_subject": val_subject, "train_subjects": train_subjects, "best_val_f1": best_val_f1,
              "n_test": len(ds["test"]), "test": protocol.evaluate(model, loader(ds["test"], False))}
    print(json.dumps(result, indent=2))
    (out_dir / f"{args.dataset_name}_{args.test_subject}.json").write_text(json.dumps(result, indent=2))
    ckpt.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
