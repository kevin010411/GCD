# GCD workspace fixes — 2026-09-30

The regression suite covers config-only layer inspection, dataset/model switching,
remembered layer selection, runtime validation of dynamic model layer declarations,
one prediction per dataset/model, independent heatmaps, prediction deletion and
recreation, preservation of prediction display settings, linked slices, duplicate
signal prevention, coalesced dragging, and interpolation parity.

Final validation: **147 tests passed** across 18 related test modules, plus AST
parsing of changed Python files and `git diff --check`.

Run the focused tests from the GCD directory in Windows 11 PowerShell using uv (after `uv sync --locked`):

```powershell
uv run python -m unittest tests.test_xai_workspace_regressions tests.test_core_engine_cam_methods tests.test_workflow_service tests.test_presenter tests.test_workspace_store tests.test_workspace_overlay tests.test_workspace_host tests.test_gradcam_plugin tests.test_xai_family_plugin tests.test_layer_hooks tests.test_model_runtime_loader tests.test_xai_cam_runner
```

The managed test process disabled YAPF grammar-cache writes in memory before
importing mmengine, because the external user cache was not writable in the
sandbox. No installed dependency files or user cache settings were changed.

## Real CT timing

`data/chgh/patient0016.nii.gz`, shape `512 × 512 × 427`, dtype `int16`, was loaded
into real Qt slice widgets in offscreen mode. The 3D renderer was not initialized.

| Workload | Comparison | New path | Repetitions |
| --- | ---: | ---: | ---: |
| Slice image refresh | All three slices: 16.66 ms | Affected slice: 4.44 ms | 12 |
| Affine plane sampling | Previous HEAD sampler: 176.21 ms | New sampler: 43.73 ms | 5, after warmup |

Times are medians on this machine. The refresh comparison isolates the number of
slice widgets refreshed; it does not measure the old 3D render scheduling cost.
The sampler comparison used a quarter-voxel translation on the existing CT. Its
maximum absolute numerical difference was `0.00586` in CT intensity units; the
small random-volume interpolation regression uses an absolute tolerance of
`2e-5`. Timings and limits are recorded in `output/workspace_fix_validation.json`.

These results establish the removed work and exercise the real Qt controls.
Desktop GPU/VTK frame rate during rotation and dragging has not been measured.
Restart GCD to use the updated code, then check layer availability while switching
datasets/configs, run multiple classes/methods against one dataset/model, and drag
each slice with the desired overlays and 3D rotation enabled.
