from __future__ import annotations

from collections.abc import Mapping

from .base import XaiLayerSelection
from .layer_gradient import LayerGradientXaiMethod
from .benchmark_cam import raw_gradient_cam


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
        from ..cam_protocol import resolve_cam_protocol
        protocol = resolve_cam_protocol(method_params.get("cam_protocol"))
        gradcam = raw_gradient_cam(activation, gradient, "gradcam", rectify=protocol["relu_stage"] == "per_tile")
        return F.interpolate(gradcam, size=selection.output_size, mode="trilinear", align_corners=False)
