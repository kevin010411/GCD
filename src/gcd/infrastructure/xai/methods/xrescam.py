from __future__ import annotations

from collections.abc import Mapping

from .base import XaiLayerSelection
from .layer_gradient import LayerGradientXaiMethod


class XResCamMethod(LayerGradientXaiMethod):
    id = "xrescam"
    display_name = "XResCAM"

    def _build_tile_cam(
        self,
        patch_payload: dict[str, object],
        selection: XaiLayerSelection,
        method_params: Mapping[str, object],
    ):
        import torch
        import torch.nn.functional as F

        activation, gradient = self._layer_tensors(patch_payload, selection.layer)
        xrescam = torch.sum(
            (activation * gradient)[:, selection.n1 : selection.n2, ...],
            dim=1,
            keepdim=True,
        )
        return F.interpolate(xrescam, size=selection.output_size, mode="trilinear")
