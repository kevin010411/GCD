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

- A compatible GPU with CUDA support is recommended for faster processing.
- NIfTI files (`.nii` or `.nii.gz`) for input data.
- Pre-trained model checkpoint file (`unetcnx.pth`).

## Quick Start

This project uses uv for dependency management.Make sure uv is installed before proceeding:https://docs.astral.sh/uv/

1. Sync project
```bash
uv sync
```
2. Load checkpoint by adjust config
all config is under the ./src/config/model，for example unet_3d,can easily edit ckpt to change your checkpoint dir
```python
_base_ = ["../base.py"]

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
```bash
uv run main.py
```


## Usage

### Dataset inference CLI

From the repository root, pass the dataset directory as the positional input:

```powershell
uv run python -m experiments.predict data/chgh `
  --config experiments/configs/predict.py `
  --output output/chgh
```

Input volumes such as `patient0001.nii.gz` are automatically paired with
`patient0001_gt.nii.gz`. Each patient is written to a separate directory under
`output/chgh`, and the dataset-level index is written to
`output/chgh/dataset_summary.json`.

For all experiment CLI options and output files, see
[`experiments/README.md`](experiments/README.md).

1. Run the application:
   ```bash
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

- The application assumes a model input size of 128x128x128 and specific spacing (`(0.7, 0.7, 1.0)`). Adjust these in `gcd_core.py` if needed.
- Video recording requires sufficient disk space and may take time depending on the rotation speed and number of frames.
- Public datasets are available to test run. For example, https://www.kaggle.com/datasets/rajendrakpandey/mm-whs-2017-dataset-5-62-gb-158-files-ct-and-mr

## Troubleshooting

- **NIfTI file errors**: Verify that input files are valid NIfTI files and not corrupted.
- **Performance issues**: Use a CUDA-enabled GPU for faster processing. Check console output for errors.
- **GUI rendering issues**: Ensure VTK and PyQt6 are correctly installed and compatible with your Python version.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

