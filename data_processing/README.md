# EMG-FoundBench data processing

Preprocessing code for the EMG-FoundBench datasets. All datasets are processed with the same
pipeline (`emg_pipeline.py`), following Sec. 3.2 (Data Processing) and Appendix B.4 (Unified
Preprocessing Pipeline) of the paper.

## Layout

| Path | Content |
|---|---|
| `emg_pipeline.py` | Shared preprocessing pipeline |
| `loaders/` | Raw-data readers, one per dataset |
| `build_benchmark.py` | Builds a classification dataset |
| `build_emg2qwerty.py` | Builds emg2qwerty |
| `build_loso.py` | Builds the leave-one-subject-out (LOSO) version of a dataset |

## Pipeline

| Paper (Sec. 3.2 / App. B.4) | Code |
|---|---|
| All recordings are mapped to a common 1000-Hz grid using polyphase resampling | `condition()` |
| Recordings at or above 1000 Hz are resampled using polyphase anti-aliasing and then filtered with a 4th-order Butterworth bandpass filter at 20–450 Hz | `condition()` |
| Recordings below 1000 Hz are bandpass filtered at the native sampling rate, with the upper cutoff constrained by the original Nyquist frequency, before upsampling | `condition()` |
| 50/60-Hz notch filter (Q = 30) where the frequency is supported by the original sampling rate | `condition()` |
| Per-channel, per-window (instance-level) Z-score normalization | `zscore()` |
| Non-overlapping 3-second windows for all downstream tasks | `WINDOW_SAMPLES` |
| Recordings are partitioned at the trial or repetition level before windowing; without trial identifiers, recordings are divided into contiguous 70/15/15 blocks | `process_subject()` |
| Pool A (~90% of subjects) and Pool B (~10%) | `assign_pools()` |
| LOSO: every subject in the cohort is held out once; datasets with more than 10 subjects use the first 10 subjects | `build_loso.py` |
| emg2qwerty: consecutive non-overlapping 3-second windows; keypresses within each window form the target sequence | `build_emg2qwerty.py` |
