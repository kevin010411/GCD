from __future__ import annotations

from collections.abc import Mapping

from .base import CamPatchContext, XaiLayerSelection, XaiMethod, XaiParameterSpec


class PerturbationOcclusionMethod(XaiMethod):
    id = "perturb_occlusion"
    display_name = "Occlusion"
    family = "perturbation"
    uses_layer_controls = True
    uses_objective = False

    def parameter_schema(self) -> tuple[XaiParameterSpec, ...]:
        return (
            XaiParameterSpec(
                "block_size",
                "Block Size",
                "int",
                default=16,
                min_value=1,
                max_value=256,
                step=1,
            ),
            XaiParameterSpec(
                "stride",
                "Stride",
                "int",
                default=8,
                min_value=1,
                max_value=256,
                step=1,
            ),
        )

    def collect_patch_data(self, context: CamPatchContext) -> dict[str, object]:
        return {
            "method": self.id,
            "pred": context.logits.detach().to("cpu"),
            "layers": {
                key: {
                    "activation": value.detach().to("cpu"),
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

        layers = patch_payload["layers"]
        if not isinstance(layers, dict) or selection.layer not in layers:
            raise KeyError(f"layer '{selection.layer}' 不存在於 CAM payload 中。")
        layer_payload = layers[selection.layer]
        if not isinstance(layer_payload, dict):
            raise TypeError("CAM layer payload 格式錯誤。")
        activation = layer_payload["activation"]
        if not isinstance(activation, torch.Tensor):
            raise TypeError("CAM layer payload 缺少 activation tensor。")
        occlusion_map = torch.mean(
            torch.abs(activation[:, selection.n1 : selection.n2, ...]),
            dim=1,
            keepdim=True,
        )
        occlusion_map = F.interpolate(
            occlusion_map, size=selection.output_size, mode="trilinear"
        )
        block_size = int(method_params.get("block_size", 16) or 16)
        kernel = max(1, min(block_size // 8, 7))
        if kernel > 1:
            occlusion_map = F.avg_pool3d(
                occlusion_map,
                kernel_size=kernel,
                stride=1,
                padding=kernel // 2,
            )
        return occlusion_map
