# Single-file prediction experiment

Run from the repository root:

```powershell
uv run python -m experiments.predict data/patient0016.nii.gz `
  --config experiments/configs/predict.py `
  --output output/patient0016_pred.nii.gz
```

Add a ground-truth label to calculate per-class and mean Dice/IoU:

```powershell
uv run python -m experiments.predict data/patient0016.nii.gz `
  --config experiments/configs/predict.py `
  --ground-truth data/patient0016_label.nii.gz `
  --output output/patient0016_pred.nii.gz
```

MMEngine overrides are supported without editing the config:

```powershell
uv run python -m experiments.predict data/patient0016.nii.gz `
  --config experiments/configs/predict.py `
  --output output/patient0016_pred.nii.gz `
  --cfg-options inference.device=cpu inference.overlap=0.5
```

The CLI writes a NIfTI prediction, one normalized `*.xai.nii.gz` attribution
volume per configured XAI method, and a sibling `*.metrics.json`. Benchmark
metrics include latency, throughput, parameter count, parameter/checkpoint size,
prediction size, and per-class voxel counts. Dice/IoU are included only when a
ground-truth volume is provided. The `xai` section contains the target class,
attribution path, insertion/deletion curves, and trapezoidal AUC values.
Gradient attribution is evaluated at the configured inference ROI resolution
and resampled to the original preprocessed volume shape, keeping 3-D memory use
bounded for large CT scans.

XAI is configured in `experiments/configs/predict.py`. Available methods are
`saliency_map`, `input_x_gradient`, and `smoothgrad`; multiple methods can be
evaluated in one run. For a faster smoke test, reduce the perturbation steps:

```powershell
uv run python -m experiments.predict data/patient0016.nii.gz `
  --config experiments/configs/predict.py `
  --output output/patient0016_pred.nii.gz `
  --cfg-options metrics.perturbation_steps=2 xai.methods="('saliency_map',)"
```
