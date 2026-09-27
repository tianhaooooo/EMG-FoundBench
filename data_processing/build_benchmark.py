#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from emg_pipeline import (assign_pools, pipeline_config, process_subject, save_subject,
                          write_json)
from loaders import registry

DEFAULT_OUT_ROOT = Path(os.environ.get("EMG_OUT_ROOT", Path(__file__).resolve().parent / "processed")) / "benchmark"


def main():
    loaders = registry()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(loaders))
    p.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--subjects", default=None, help="comma-separated subset, e.g. s1,s2")
    args = p.parse_args()

    loader = loaders[args.dataset]()
    out_dir = args.out_root / args.dataset
    all_subjects = loader.subjects()
    subjects = args.subjects.split(",") if args.subjects else all_subjects
    print(f"{args.dataset}: {len(subjects)}/{len(all_subjects)} subjects -> {out_dir}", flush=True)

    counts, empty = {}, []
    for s in subjects:
        result = process_subject(loader.load(s), loader.classes, loader.short_trial_handling)
        if sum(len(y) for _, y in result.values()) == 0:
            empty.append(s)
            print(f"  {s}: no windows", flush=True)
            continue
        counts[s] = save_subject(out_dir, s, result)
        missing = {sp: sorted(set(result["train"][1].tolist()) - set(y.tolist()))
                   for sp, (_, y) in result.items() if sp != "train"}
        print(f"  {s}: {counts[s]}" + (f"  classes absent from val/test: {missing}"
                                         if any(missing.values()) else ""), flush=True)

    write_json(out_dir / "dataset_info.json", dict(
        dataset=args.dataset, classes=[str(c) for c in loader.classes],
        n_classes=len(loader.classes), pipeline=pipeline_config(),
        subjects=counts, subjects_without_windows=empty))
    write_json(out_dir / "pool_assignment.json", assign_pools(list(counts)))
    print(f"done: {len(counts)} subjects written, {len(empty)} without windows")


if __name__ == "__main__":
    main()
