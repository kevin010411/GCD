from __future__ import annotations

from collections.abc import Mapping

from .base import XaiLayerSelection
from .layer_gradient import LayerGradientXaiMethod
from .benchmark_cam import raw_gradient_cam


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
        from ..cam_protocol import resolve_cam_protocol
        rectify = resolve_cam_protocol(method_params.get("cam_protocol"))["relu_stage"] == "per_tile"
        xrescam = raw_gradient_cam(
            activation[:, selection.n1 : selection.n2],
            gradient[:, selection.n1 : selection.n2], "hirescam", rectify=rectify,
        )
        return F.interpolate(xrescam, size=selection.output_size, mode="trilinear", align_corners=False)


class HiResCamMethod(XResCamMethod):
    id = "hirescam"
    display_name = "HiResCAM"


class LayerCamMethod(XResCamMethod):
    id = "layercam"
    display_name = "LayerCAM"

    def _build_tile_cam(self, patch_payload, selection, method_params):
        import torch.nn.functional as F
        activation, gradient = self._layer_tensors(patch_payload, selection.layer)
        from ..cam_protocol import resolve_cam_protocol
        rectify = resolve_cam_protocol(method_params.get("cam_protocol"))["relu_stage"] == "per_tile"
        cam = raw_gradient_cam(activation[:, selection.n1:selection.n2],
                               gradient[:, selection.n1:selection.n2], "layercam", rectify=rectify)
        return F.interpolate(cam, size=selection.output_size, mode="trilinear", align_corners=False)
