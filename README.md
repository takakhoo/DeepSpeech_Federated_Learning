# ASR Grad Reconstruction DS1 – Long-Audio Experiments

This subdirectory extends Minh's DS1 gradient-matching pipeline with tooling
for ≥10 s utterances and concatenation experiments while keeping the original
scripts intact.

## Timeline (UTC)

- **2025-11-21 02:40** – Added `src/data/build_long_audio_dataset.py` to
  concatenate existing 0–4 s HDF5 waveforms into long-form samples with
  per-segment metadata.
- **2025-11-21 02:47** – Generated 140 synthetic ≥10 s samples:
  ```
  python src/data/build_long_audio_dataset.py \
    --source-dir /scratch2/f004h1v/datasets/librispeech_sampled_600_file_0s_4s \
    --output-dir /scratch2/f004h1v/datasets/librispeech_concat_10s \
    --target-duration-ms 10000 --min-trailing-ms 8000 --max-samples 200
  ```
- **2025-11-21 02:48** – Introduced `src/data/long_audio_dataset.py` and
  `src/run_long_audio_reconstruction.py` to (a) expose metadata-aware dataloaders
  and (b) run chunked reconstructions that preserve Minh’s loss functions.
- **2025-11-21 02:58** – Ran the first ≥10 s reconstruction (sample `0`) with
  600 first-order + 60 zero-order iterations per chunk:
  ```
  cd federated_learning/asr_grad_reconstruction_ds1/src
  CUDA_VISIBLE_DEVICES=0 python run_long_audio_reconstruction.py \
    --batch-start 0 --batch-end 1 \
    --dataset_path /scratch2/f004h1v/datasets/librispeech_concat_10s \
    --max_iter 600 --zero_max_iter 60 --lr 0.4 --zero_lr 15
  ```
  - Total runtime: ~868 s to reconstruct 21 segments (≈10.2 s audio).
  - Aggregate metrics (`sampleidx_0_concat.pt`): **MSE 26.17**, **MAE 3.73**,
    **PSNR 23.55 dB**.
  - Artifacts: `long_audio_runs/10s_plus/ds1_lr0.4_iter600_start0_end1/`
    contains per-chunk spectrograms, tensors, per-chunk MAE stats, and a stitched summary figure.
- **2025-11-21 03:10** – Added `src/tools/report_long_audio_metrics.py` to
  summarize runs (CSV + PSNR/runtime plot) and upgraded the concatenation
  figure to include per-chunk MAE bars for easier interpretation.

## Usage Notes

### 1. Create ≥10 s Dataset

```
python src/data/build_long_audio_dataset.py \
  --source-dir <short-h5-dir> \
  --output-dir <long-h5-dir> \
  --target-duration-ms 10000
```

Each output HDF5 stores the concatenated waveform, transcript, duration, and
per-segment metadata (`segment_text`, `segment_durations`, `segment_file_paths`).

### 2. Chunked Reconstruction Runner

```
cd federated_learning/asr_grad_reconstruction_ds1/src
CUDA_VISIBLE_DEVICES=<gpu> python run_long_audio_reconstruction.py \
  --dataset_path /path/to/librispeech_concat_10s \
  --batch-start 0 --batch-end 2 \
  --max_iter 600 --zero_max_iter 60 --lr 0.4 --zero_lr 15
```

For each long utterance:

1. Split MFCCs according to stored segment durations.
2. Run Minh’s DS1 optimizer per chunk (first-order + optional zero-order).
3. Save chunk tensors (`sampleidx_<i>_chunk_<j>_x_param_last.pt`).
4. Concatenate reconstructed MFCCs and emit figures + metrics under
   `long_audio_runs/<bucket>/`.

Original scripts (`reconstruct_ds1_run_many_sample.py`, etc.) remain untouched,
so we can fall back to the short-duration workflow at any time.

### 3. Summarize Metrics

```
cd federated_learning/asr_grad_reconstruction_ds1/src
python tools/report_long_audio_metrics.py --runs-dir ../long_audio_runs/10s_plus
```

The tool writes `metrics_summary.csv` and `metrics_summary.png` next to your
experiment folders. Each row includes runtime, segment count, duration, MSE,
MAE, and PSNR. The plot overlays PSNR and runtime across utterances.

## Interpreting Outputs

- **Chunk PNGs** (`figures/chunk_<id>_*.png`): Top-left = target MFCC, top-right =
  reconstruction, bottom-left = absolute difference heatmap, bottom-right =
  loss traces. Loss curves look similar because each chunk optimizes the same
  cosine-gradient objective (same scale) and follows the same LR schedule;
  variation appears mainly in convergence rate / final value.
- **`sampleidx_*_concat.png`**: stitched spectrograms comparing the full
  reconstructed utterance to the ground truth with aggregate difference.
- **`.pt` files**: tensors + metadata for quantitative analysis. Load them with
  PyTorch to compute metrics or to regenerate waveforms downstream.

For a “good” reconstruction, expect decreasing loss, sharper spectrogram
alignment, PSNR > ~20 dB, and MAE trending downwards. When concatenating longer
clips, check both the per-chunk PNGs (fine detail) and the aggregate metrics
from `report_long_audio_metrics.py` (`metrics_summary.csv` + `.png`) to
summarize performance across runs. Example (sampleidx_0 from
`ds1_lr0.4_iter600_start0_end1`):

| duration (s) | segments | runtime (s) | MSE | MAE | PSNR (dB) | max chunk MAE |
|-------------|----------|-------------|-----|-----|-----------|----------------|
| 10.22       | 21       | 868.3       | 27.18 | 3.79 | 23.35 | 3.85 |

Having multiple run folders (e.g., the older `ds1_lr0.4_iters600_sample0_old`
and the refreshed `ds1_lr0.4_iter600_start0_end1`) now produces multiple
entries/points on the summary plot, clarifying how PSNR and runtime compare
across experiments.

### 4. Share-ready Figures

For GitHub-friendly reporting we keep a light-weight gallery in
`long_audio_runs/10s_plus/ds1_lr0.4_iter600_start0_end1/share_figures/`.
It currently contains:

- `sampleidx_0_chunk_[0-9]*.png`: the first ten chunk visualizations (target vs
  reconstruction vs difference).
- `sampleidx_0_concat_summary_*.png`: full-utterance comparison with per-chunk
  MAE bar chart.

Only this folder plus `metrics_summary.csv` need to be tracked when pushing
results upstream.

### 5. GitHub Push Checklist

1. Stage code + docs:
   ```
   git add federated_learning/asr_grad_reconstruction_ds1/src/data/build_long_audio_dataset.py \
           federated_learning/asr_grad_reconstruction_ds1/src/data/long_audio_dataset.py \
           federated_learning/asr_grad_reconstruction_ds1/src/run_long_audio_reconstruction.py \
           federated_learning/asr_grad_reconstruction_ds1/src/tools/report_long_audio_metrics.py \
           federated_learning/asr_grad_reconstruction_ds1/README.md \
           federated_learning/README.md
   ```
2. Stage shareable artifacts:
   ```
   git add federated_learning/asr_grad_reconstruction_ds1/long_audio_runs/10s_plus/metrics_summary.csv \
           federated_learning/asr_grad_reconstruction_ds1/long_audio_runs/10s_plus/ds1_lr0.4_iter600_start0_end1/share_figures
   ```
3. Commit & push:
   ```
   git commit -m "Add long-audio DS1 runner, metrics tooling, and sample0 results"
   git push origin main
   ```

> Note: the repository currently has pre-existing deletions under
> `asr-grad-reconstruction/` (Minh’s older DS2 work). Do **not** stage those
> deletions unless you intentionally removed that directory. Use
> `git restore --staged asr-grad-reconstruction` if necessary before committing.
