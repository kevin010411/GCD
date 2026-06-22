from __future__ import annotations

from collections.abc import Mapping

from .base import XaiLayerSelection
from .layer_gradient import LayerGradientXaiMethod


class GradCamMethod(LayerGradientXaiMethod):
    id = "gradcam"
    display_name = "Grad-CAM"

    def _build_tile_cam(
        self,
        patch_payload: dict[str, object],
        selection: XaiLayerSelection,
        method_params: Mapping[str, object],
    ):
        import torch
        import torch.nn.functional as F

        activation, gradient = self._layer_tensors(patch_payload, selection.layer)
        activation = activation[:, selection.n1 : selection.n2, ...]
        gradient = gradient[:, selection.n1 : selection.n2, ...]
        weights = torch.mean(gradient, dim=(2, 3, 4), keepdim=True)
        gradcam = torch.sum(activation * weights, dim=1, keepdim=True)
        return F.interpolate(gradcam, size=selection.output_size, mode="trilinear")
