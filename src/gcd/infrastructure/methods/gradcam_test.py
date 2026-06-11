from __future__ import annotations

from collections.abc import Mapping

from .base import CamPatchContext, XaiLayerSelection
from .layer_gradient import LayerGradientXaiMethod


class GradCAMTestMethod(LayerGradientXaiMethod):
    id = "gradcam_test"
    display_name = "Grad-CAM Test"
    family = "gradient"
    uses_layer_controls = True
    uses_objective = False

    def __init__(self) -> None:
        pass

    @staticmethod
    def _objective(logits, target_class: int):
        if not (0 <= target_class < logits.size(1)):
            raise ValueError(
                f"target_class={target_class} 超出模型輸出範圍 0..{logits.size(1) - 1}"
            )
        return logits[0, target_class].sum()

    def collect_patch_data(self, context: CamPatchContext) -> dict[str, object]:
        loss = self._objective(context.logits, context.target_class)
        loss.backward()
        return {
            "method": self.id,
            "pred": context.logits.detach().to("cpu"),
            "layers": {
                key: {
                    "activation": value.detach().to("cpu"),
                    "gradient": self._required_gradient(value),
                }
                for key, value in context.layers_by_name.items()
            },
        }

    def _build_tile_cam(
        self,
        patch_payload: dict[str, object],
        selection: XaiLayerSelection,
        method_params: Mapping[str, object],
    ):
        import torch
        import torch.nn.functional as F

        activation, gradient = self._layer_tensors(patch_payload, selection.layer)
        test_cam = torch.sum(
            (activation * gradient)[:, selection.n1 : selection.n2, ...],
            dim=1,
            keepdim=True,
        )
        return F.interpolate(test_cam, size=selection.output_size, mode="trilinear")
