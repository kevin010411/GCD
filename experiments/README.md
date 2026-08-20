# Single-file prediction experiment

Run from the repository root:

```powershell
uv run python -m experiments.predict data/patient0016.nii.gz `
  --config experiments/configs/predict.py `
  --output output/patient0016
```

Add a ground-truth label to calculate per-class and mean Dice/IoU:

```powershell
uv run python -m experiments.predict data/patient0016.nii.gz `
  --config experiments/configs/predict.py `
  --ground-truth data/patient0016_gt.nii.gz `
  --output output/patient0016
```

MMEngine overrides are supported without editing the config:

```powershell
uv run python -m experiments.predict data/patient0016.nii.gz `
  --config experiments/configs/predict.py `
  --output output/patient0016 `
  --cfg-options inference.device=cpu inference.overlap=0.5
```

`--output` is a result directory. The CLI shows tqdm progress bars with ETA and
writes the following analysis bundle inside it:

```text
patient0016/
├── patient0016_prediction.nii.gz
├── patient0016_<method>_xai.nii.gz
├── metrics.json
├── summary.csv
├── benchmark_runs.csv
├── class_voxels.csv
├── segmentation_metrics.csv       # only with --ground-truth
├── xai_summary.csv
├── xai_curves.csv
└── <method>_insertion_deletion.png # when either curve is enabled
```

Benchmark
metrics include latency, throughput, parameter count, parameter/checkpoint size,
prediction size, and per-class voxel counts. Dice/IoU are included only when a
ground-truth volume is provided. The `xai` section contains the target class,
attribution path, insertion/deletion curves, and trapezoidal AUC values.
Gradient attribution is evaluated at the configured inference ROI resolution
and resampled to the original preprocessed volume shape, keeping 3-D memory use
bounded for large CT scans.

XAI is configured in `experiments/configs/predict.py` and resolves methods from
the same `XaiMethodRegistry` used by the GUI. This includes `gradcam`, `xrescam`,
`saliency_map`, `perturb_occlusion`, `perturb_lime`, and `perturb_rise`;
multiple methods can be evaluated in one run. For a faster smoke test, reduce
the perturbation steps:

```powershell
uv run python -m experiments.predict data/patient0016.nii.gz `
  --config experiments/configs/predict.py `
  --output output/patient0016 `
  --cfg-options metrics.perturbation_steps=2 xai.methods="('saliency_map',)"
```
