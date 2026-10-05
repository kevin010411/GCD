# Grad-CAM Discoverer
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

[![Watch the video](https://img.youtube.com/vi/uvZO47YlpQk/0.jpg)](https://youtu.be/uvZO47YlpQk?si=uhN42inmI4op33RE)  
*Click the image above to watch a system demonstration on YouTube.*

Grad-CAM Discoverer is a Python application for visualizing 3D medical imaging data (e.g., CT scans) using Grad-CAM (Gradient-weighted Class Activation Mapping) with a PyQt6-based GUI and VTK for volume rendering. It allows users to load NIfTI files, process them with a pre-trained model, visualize the results with customizable transfer functions, and interact with the visualization through rotation controls and feature selection.

## Table of Contents
- [Grad-CAM Discoverer](#grad-cam-discoverer)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Prerequisites](#prerequisites)
  - [Quick Start](#quick-start)
  - [Usage](#usage)
  - [Notes](#notes)
  - [Troubleshooting](#troubleshooting)
  - [License](#license)

## Features

- **Load and Process NIfTI Files**: Load medical imaging data in NIfTI format (`.nii` or `.nii.gz`) and process it using a pre-trained UNETCNX model.
- **Grad-CAM Visualization**: Generate and display Grad-CAM heatmaps overlaid on 3D olume data.
- **Interactive GUI**: Built with PyQt6, featuring:
  - File selection for loading NIfTI files.
  - Transfer function editor for customizing color and opacity.
  - Explainability method selector with an extensible gradient-based CAM workflow.
  - Layer and feature selection for controlling Grad-CAM outputs.
  - Rotation speed control for 3D visualization.
  - Screenshot and video recording capabilities.
- **VTK Rendering**: High-quality 3D volume rendering with axes indicators and smooth rotation.
- **Console Output**: Real-time feedback on processing steps and errors.

## Prerequisites

- Windows 11 with PowerShell and uv installed.
- Python 3.11 (the project version in `.python-version`; uv manages the environment).
- A compatible GPU with CUDA support is recommended for faster processing.
- NIfTI files (`.nii` or `.nii.gz`) for input data.
- Pre-trained model checkpoint file (`unetcnx.pth`).

## Quick Start

Run this project natively on Windows 11 using PowerShell and uv. Install uv using the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/).

1. Open PowerShell and sync dependencies from the project root (adjust the checkout path if needed):
```powershell
Set-Location D:\KevinFu\GCD
uv sync --locked
```
2. Load checkpoint by adjust config
all config is under the ./config/model，for example unet_3d,can easily edit ckpt to change your checkpoint dir
```python
_base_ = ["../preprocessing/gcd.py"]

model = dict(
    type="UNet",
    act="RELU",
    norm="BATCH",
    out_channels=4,
)  # 模型
# ckpt = "checkpoint/3d_unet_2025.pth"  # checkpoint
ckpt = "checkpoint/3d_unet_60_20_20.pth"  # checkpoint
default_layer = "decoder 1"  # default layer of CAM 
```
3. Start 
```powershell
uv run main.py
```


## Shared configuration

Model configs inherit `config/preprocessing/gcd.py` or `acdc.py` directly;
`config/base.py` and top-level `size`, `stride`, `spacing`, `permute` are removed.
Set `inference.roi_size` and `inference.overlap` for every tiled XAI method,
`preprocessing.spacing` for resampling, and `display.permute` for display axes.
Padding uses the configured ROI, not the old size-plus-stride four-tile canvas.
`legacy` / `legacy_four_tile` strategies are unsupported and rejected. Results
from the old four-tile/padding protocol require rerunning before comparison.

## Usage

### Experiment CLI

Only two public experiment entrypoints remain. Run them from the GCD root in Windows 11 PowerShell using uv:

```powershell
uv run python -m experiments.benchmark --help
uv run python -m experiments.evaluate_metrics --help
```

Experiment implementation and batch configs are under `experiments/src/`;
shared model/preprocessing configs remain under root `config/`.
The old `experiments.predict` and `experiments.xai_benchmark` command paths
are removed; use the two entrypoints above.

For all experiment CLI options and output files, see
[`experiments/README.md`](experiments/README.md).

1. Run the application:
   ```powershell
   uv run main.py
   ```

2. The GUI will open with the following controls:
   - **Open File**: Select a NIfTI file from the `dat` directory or elsewhere.
   - **Method Selection**: Choose the explainability method. The current implementation ships with Grad-CAM and keeps the workflow extensible for future gradient-based methods.
   - **Transfer Function Editor**: Click and drag to adjust control points for color and opacity. Double-click to change colors.
   - **Layer Selection**: Choose a layer from the model to compute Grad-CAM.
   - **Feature Selection**: Adjust the feature range using input fields or shift buttons (`<` and `>`).
   - **Rotation Controls**: Use the slider to adjust rotation speed or start/stop rotation.
   - **Overlay/Heatmap**: Toggle between overlay mode (heatmap + CT) and heatmap-only mode.
   - **Save Screenshot**: Save the current view as a PNG file.
   - **Record Video**: Record a video of the visualization as an MP4 file.

## Notes

- GradCAM, HiResCAM (also available as the legacy XResCAM name), LayerCAM and ScoreCAM use full sliding-window coverage with the configured ROI, overlap and blending. The default predicted-target objective uses a fixed full-volume prediction mask; gradient CAMs differentiate the mean target logit, rectify each raw tile before fusion, and normalize only the final volume. Benchmark CAM formulas live in `src/gcd/infrastructure/xai/methods/benchmark_cam.py`.
- Model input size, stride, spacing, and display axis order come from the selected `config/model/*.py` config.
- Layer options are discovered by constructing the configured model and reading its `xai_layer_targets`; configs do not need a duplicate layer list. One evaluation-mode, inference-only forward using the configured window size measures each layer's output channels. Only names and channel counts are cached, and temporary hooks/model tensors are released. Checkpoint loading remains part of XAI execution, not layer inspection.
- Switching datasets keeps the selected model's layers available. Remembered layer selections are scoped to the dataset and model, and feature counts from another model are discarded.
- Each dataset has one prediction volume per model architecture and checkpoint revision, named `<dataset>_<model>_prediction`. Repeated XAI runs update that prediction while preserving its name, visibility, and transfer function. Each heatmap remains a separate volume. Different datasets or checkpoints retain their own predictions.
- Slice dragging emits one change per slider/spinbox update, coalesces consecutive changes with a 16 ms timer, and refreshes only the changed slice and linked slices with the same orientation. It does not request extra 3D renders. Overlay affine sampling reads only the current plane, and overlay color maps are prepared when the workspace payload changes.
- Video recording requires sufficient disk space and may take time depending on the rotation speed and number of frames.
- Public datasets are available to test run. For example, https://www.kaggle.com/datasets/rajendrakpandey/mm-whs-2017-dataset-5-62-gb-158-files-ct-and-mr

## Troubleshooting

- **NIfTI file errors**: Verify that input files are valid NIfTI files and not corrupted.
- **Performance issues**: Use a CUDA-enabled GPU for faster processing. Check console output for errors.
- **GUI rendering issues**: Ensure VTK and PyQt6 are correctly installed and compatible with your Python version.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

