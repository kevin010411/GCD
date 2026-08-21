# Single-file prediction experiment

Pass a directory to infer every `.nii`/`.nii.gz` volume in it. Files ending in
`_gt` are automatically paired with the corresponding input:

```powershell
uv run python -m experiments.predict data/chgh `
  --config experiments/configs/predict.py `
  --output output/chgh
```

Each case is written to its own subdirectory and `dataset_summary.json` is
written at the dataset output root.

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
├── patient0016_<method>_class_<class>_xai.nii.gz
├── metrics.json
├── summary.csv
├── benchmark_runs.csv
├── class_voxels.csv
├── segmentation_metrics.csv       # only with --ground-truth
├── xai_summary.csv
├── xai_curves.csv
└── plots/
    └── <method>/
        └── class_<class>/
            ├── insertion_0/
            │   ├── insertion_<class>_target_probability.png
            │   ├── insertion_<class>_prediction_dice.png
            │   ├── insertion_<class>_prediction_iou.png
            │   ├── insertion_<class>_ground_truth_dice.png
            │   └── insertion_<class>_ground_truth_iou.png
            └── deletion_0/
                └── deletion_<class>_<metric>.png
```

Each metric is written to its own figure. The config index is represented by
the parent directory (`insertion_0`, `insertion_1`, and so on), preventing
different configs for the same class from overwriting one another.

Benchmark
metrics include latency, throughput, parameter count, parameter/checkpoint size,
prediction size, and per-class voxel counts. Dice/IoU are included only when a
ground-truth volume is provided. The `xai` section contains the target class,
attribution path, independently configured perturbation curves, and AUC values.
Gradient attribution is evaluated at the configured inference ROI resolution
and resampled to the original preprocessed volume shape, keeping 3-D memory use
bounded for large CT scans.

XAI is configured in `experiments/configs/predict.py` and resolves methods from
the same `XaiMethodRegistry` used by the GUI. This includes `gradcam`, `xrescam`,
`saliency_map`, `perturb_occlusion`, `perturb_lime`, and `perturb_rise`.

Insertion and deletion are configured as lists named `PerturbationInsertion`
and `PerturbationDeletion`. Each list may contain any number of independent
config dictionaries; an empty list disables that operation. Every dictionary
has its own `method`, `target_class`, `steps`, `baseline`, and
`answer_retention`, and method/class may themselves be lists. An empty method
or target-class list skips that dictionary.

`answer_retention=(None, 0.5)` produces both a standard curve and an
answer-preserving curve where at least 50% of target-class GT voxels remain
unchanged. Numeric variants need ground truth; without it they are recorded as
skipped while the standard (`None`) variant still runs.

For a faster run, reduce the perturbation steps:

```powershell
uv run python -m experiments.predict data/patient0016.nii.gz `
  --config experiments/configs/predict.py `
  --output output/patient0016 `
  --cfg-options PerturbationInsertion.0.steps=2 PerturbationDeletion.0.steps=2
```
