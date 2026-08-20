from __future__ import annotations

from collections.abc import Mapping

from .base import CamPatchContext, XaiLayerSelection, XaiMethod


class OrganOcclusionMethod(XaiMethod):
    """Dataset-level semantic intervention registered in the perturbation family.

    The regular patch methods are intentionally minimal: the dataset-level runner
    applies organ masks before inference and only reuses these hooks to obtain a
    fused prediction from GCD's existing tiling pipeline.
    """

    id = "organ_occlusion"
    display_name = "Organ Occlusion"
    family = "perturbation"
    uses_layer_controls = False
    uses_objective = False
    execution_scope = "dataset"
    result_kind = "intervention_comparison"
    custom_ui = "organ_occlusion"

    def collect_patch_data(self, context: CamPatchContext) -> dict[str, object]:
        return {
            "method": self.id,
            "pred": context.logits.detach().to("cpu"),
        }

    def _build_tile_cam(
        self,
        patch_payload: dict[str, object],
        selection: XaiLayerSelection,
        method_params: Mapping[str, object],
    ):
        import torch

        return torch.zeros((1, 1, *selection.output_size), dtype=torch.float32)

