# EMG-FoundBench: foundation-model experiments

Code for the downstream evaluation of pretrained time-series and electrophysiology foundation
models on EMG (Appendix C.8 - C.10 of the paper). Every model family is trained and evaluated
with the **same protocol code** (`common/`); a family only defines how an EMG window enters its
backbone and how the backbone output is pooled (`<family>/model.py`).

## Layout

```
common/
  protocol.py      optimizer, learning rates, early stopping, evaluation, five-shot adaptation
  data.py          Pool A / Pool B split, LOSO cohort, label mapping
  ctc.py           emg2qwerty CTC interface, training loop and per-user CER
<family>/model.py  model-specific input interface, pooling, classification / CTC head, batch size
<family>/requirements.txt
models.py          registry used by the entry points (--model)
train_pool.py      linear probing / full fine-tuning / train-from-scratch on Pool A,
                   evaluated on Pool A test and (zero-calibration) Pool B test
loso.py            cohort-restricted leave-one-subject-out, one fold per call
five_shot.py       five-shot head adaptation on Pool B, seeds 42-46
emg2qwerty_ctc.py  continuous typing decoding with CTC
```

## Protocol (identical for all families)

| | |
|---|---|
| Optimizer | AdamW, betas (0.9, 0.95), weight decay 0.01, eps 1e-7, constant learning rate |
| Learning rate | 1e-5 linear probing; 6e-5 full fine-tuning and scratch |
| Budget | up to 50 epochs, early stopping on validation macro-F1 (patience 10), best-validation checkpoint |
| Precision | bfloat16 autocast |
| Linear probing | backbone frozen and kept in eval mode; only the linear head is trained |
| Scratch | same architecture, data, batch size and recipe as full fine-tuning; random initialization |
| Five-shot | start from the Pool A full fine-tuning checkpoint; 5 windows per class from Pool B train; head only (backbone frozen, eval mode), AdamW lr 5e-4, 30 epochs, final epoch evaluated on Pool B test; seeds 42-46 |
| LOSO | first 10 subjects in canonical order (all if at most 10); per fold one test subject, one validation subject, the rest for training; full fine-tuning recipe |
| emg2qwerty | native backbone input, channel representations concatenated per timestep, linear CTC head (70 characters + blank); full fine-tuning recipe with early stopping on validation CER; CER on the per-user concatenation of decoded windows |

Batch size is 32 when memory permits and is otherwise reduced to 4-16 (see `batch_size()` in each
`model.py`); paired runs share one batch size.

## Model families

| `--model` | Sizes | Weights | Classification interface |
|---|---|---|---|
| `moment` | small, base | AutonLab/MOMENT-1-{small,base} | native multichannel input, patch 8; mean over patches, concatenation across channels |
| `chronos2` | small, base | autogluon/chronos-2{-small,} | channels as one grouped multivariate series, patch 16; mean over patches, then channels |
| `timesfm` | base | google/timesfm-2.5-200m-pytorch | channel-independent, patch 32; mean over valid patches, then channels |
| `lagllama` | base | time-series-foundation-models/Lag-Llama | channel-independent lag tokens (1908 per window); mean over tokens, then channels |
| `patchtst` | base | ibm/patchtst-etth1-forecasting | channel-independent, patch 12; mean over patches and channels |
| `brant` | base | official Brant release (505.68M) | window interpolated to 15 x 1500 samples, 8-band log power; temporal then channel encoder; mean over patches and channels |
| `units` | base | official UniTS x128 checkpoint | native variate dimension with prompt / classification tokens; pooled classification feature before category matching, mean over channels |
| `bilstm` | base | none (scratch only) | depthwise conv to ~150 steps, 2-layer BiLSTM, final hidden states |

Environment variables: `LAG_LLAMA_SRC` (root of a Lag-Llama checkout, commit df7531a),
`BRANT_CKPT_DIR` (folder with `time_encoder.pt` and `channel_encoder.pt`), optional `UNITS_CKPT`.
Each family lists its tested package versions in `<family>/requirements.txt`.

## Data

Produced by `../data_processing`:

```
<dataset>/<subject>/{train,val,test}_{emg,labels}.npy       # emg: (N, C, 3000) float32
<dataset>/pool_A/<subject>/..., <dataset>/pool_B/<subject>/...  # datasets with a predefined partition
<loso_dataset>/<subject>/{emg,labels}.npy                     # LOSO layout
emg2qwerty/user_<id>/{train,val,test}_{emg,labels}.npy        # labels: target strings
```

## Examples

```bash
python train_pool.py --model chronos2 --model_size base --mode fft \
    --data_dir /path/to/NinaPro_DB1 --dataset_name NinaPro_DB1 --out_dir results/chronos2_base_fft
python five_shot.py --model chronos2 --model_size base --data_dir /path/to/NinaPro_DB1 \
    --dataset_name NinaPro_DB1 --pool_a_ckpt results/chronos2_base_fft/NinaPro_DB1_fft_best.pt \
    --out_dir results/chronos2_base_five_shot
python loso.py --model timesfm --data_dir /path/to/loso/NinaPro_DB1 --dataset_name NinaPro_DB1 \
    --test_subject s1 --out_dir results/timesfm_loso
python emg2qwerty_ctc.py --model moment --model_size small --data_root /path/to/emg2qwerty \
    --out_dir results/moment_small_emg2qwerty
python train_pool.py --model bilstm --mode scratch --data_dir /path/to/NinaPro_DB1 \
    --dataset_name NinaPro_DB1 --out_dir results/bilstm_scratch
```

Run the entry points from this folder. For the BiLSTM, pass its scratch checkpoint as
`--pool_a_ckpt` to `five_shot.py`.
