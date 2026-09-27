"""emg2qwerty keystroke decoding with CTC (see common/ctc.py for the interface and recipe).

Example:
  python emg2qwerty_ctc.py --model moment --model_size small \
      --data_root /path/to/emg2qwerty --out_dir results/moment_small_emg2qwerty
"""
import argparse

import models
from common import ctc, protocol


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(models.FAMILIES), required=True)
    p.add_argument("--model_size", choices=models.ALL_SIZES, default="base")
    p.add_argument("--data_root", required=True, help="folder with one user_* sub-folder per user")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_users", type=int, default=None, help="use only the first N users (default: all users)")
    p.add_argument("--batch_size", type=int, default=None, help="override the default batch size")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--init_ckpt", default=None, help="continue training from this checkpoint")
    p.add_argument("--start_epoch", type=int, default=1, help="epoch number to continue from")
    args = p.parse_args()

    protocol.set_seed(args.seed)
    family = models.get_family(args.model, args.model_size)
    name = models.display_name(family, args.model_size)
    bs = args.batch_size or family.CTC_BATCH_SIZE[args.model_size]
    ctc.run(family.build_ctc(args.model_size), name, args.data_root, args.n_users, bs, args.out_dir,
            num_workers=args.num_workers, init_ckpt=args.init_ckpt, start_epoch=args.start_epoch)


if __name__ == "__main__":
    main()
