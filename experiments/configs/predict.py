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

# MMEngine builds every method, high-level metric, nested scorer, and final
# answer aggregator from these configs. Runtime tensors/callbacks are supplied
# separately through XaiExecutionContext rather than stored in config.
XaiMethods = [
    # dict(
    #     type="RegistryXaiMethod",
    #     id="xrescam",
    #     method="xrescam",
    #     layer="",
    #     params=dict(),
    # ),
    dict(
        type="OrganOcclusionXaiMethod",
        id="organ_occlusion",
        target_class=[1, 2, 3],
        objective=dict(type="PredictionDiceScore"),
        segmenter=dict(
            type="TotalSegmentatorService",
            cache_root="output/organ_masks",
            task="total",
            merge_organs=False,
        ),
        occluder=dict(
            type="OrganOccluder",
            mode="local_mean",
            fill_hu=0.0,
            feather_mm=2.0,
            blur_sigma_mm=3.0,
            local_mean_shell_mm=5.0,
            preserve_answer=True,
        ),
    ),
]

common_scorers = [
    dict(type="TargetProbabilityScore"),
    dict(type="PredictionDiceScore"),
    dict(type="PredictionIoUScore"),
    dict(type="GroundTruthDiceScore"),
    dict(type="GroundTruthIoUScore"),
]

# XaiMetrics = [
#     dict(
#         type="PerturbationInsertion",
#         config_index=0,
#         target_class=[1, 2, 3],
#         steps=11,
#         methods=["organ_occlusion"],
#         baseline=0.0,
#         answer_retention=[None],
#         scorers=common_scorers,
#     ),
#     dict(
#         type="PerturbationDeletion",
#         config_index=0,
#         target_class=[1, 2, 3],
#         steps=11,
#         methods=["organ_occlusion"],
#         baseline=0.0,
#         answer_retention=[None],
#         scorers=common_scorers,
#     ),
# ]

XaiAnswer = dict(
    type="FaithfulnessAnswerAggregator",
    insertion_weight=0.5,
    deletion_weight=0.5,
    curve="target_probability",
    variant="standard",
    insertion_metric="insertion_0",
    deletion_metric="deletion_0",
)
