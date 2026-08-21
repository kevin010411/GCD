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
)

# Each list accepts any number of independent configs; an empty list disables
# that operation. Empty method/target_class lists also skip one config. ``None``
# runs the standard curve; numeric values additionally keep at least that
# fraction of the class GT untouched at every step.
PerturbationInsertion = [
    dict(
        target_class=[1, 2, 3],
        steps=11,
        method=["xrescam"],
        baseline=0.0,
        answer_retention=[None, 0.5, 1.0],
    ),
]

PerturbationDeletion = [
    dict(
        target_class=[1, 2, 3],
        steps=11,
        method=["xrescam"],
        baseline=0.0,
        answer_retention=[None, 0.5, 1.0],
    ),
]

xai = dict(
    enabled=True,
    layer="",
    method_params=dict(),
)
