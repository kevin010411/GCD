_base_ = ["../../src/config/model/inception_resblock.py"]

# Single-volume segmentation inference and benchmark settings.
inference = dict(
    device="auto",  # auto, cpu, cuda, or cuda:0
    roi_size=(128, 128, 128),
    sw_batch_size=1,
    overlap=0.25,
    blend_mode="constant",  # constant or gaussian
    warmup_runs=0,
    benchmark_runs=1,
)

preprocessing = dict(
    spacing=(0.7, 0.7, 1.0),
    intensity_input_range=(-42.0, 423.0),
    intensity_output_range=(0.0, 1.0),
)

metrics = dict(
    include_background=False,
    empty_score=1.0,
    # Insertion/deletion evaluate 0%, 10%, ..., 100% of ranked voxels.
    perturbation_steps=10,
    perturbation_baseline=0.0,
)

xai = dict(
    enabled=True,
    # Supported: saliency_map, input_x_gradient, smoothgrad.
    methods=("saliency_map",),
    # "auto" selects the largest predicted foreground class.
    target_class="auto",
    smoothgrad_samples=8,
    smoothgrad_noise_std=0.05,
)
