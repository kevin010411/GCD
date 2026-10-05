from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from .base import CamPatchContext, XaiLayerSelection, XaiMethod

if TYPE_CHECKING:
    import torch


class ScoreCamMethod(XaiMethod):
    """Gradient-free Score-CAM for 3D segmentation feature maps."""

    id = "scorecam"
    display_name = "Score-CAM"
    family = "gradient"
    uses_layer_controls = True
    uses_objective = True

    def __init__(self, objective: Callable[[torch.Tensor, int], torch.Tensor]) -> None:
        self._objective = objective

    def collect_patch_data(self, context: CamPatchContext) -> dict[str, object]:
        import torch
        import torch.nn.functional as F

        if context.model is None:
            raise RuntimeError("Score-CAM 需要可執行的模型。")

        params = dict(context.method_params or {})
        from ..cam_protocol import resolve_cam_protocol, target_score
        protocol = resolve_cam_protocol(params.get("cam_protocol"))
        requested_layer = str(params.get("_selected_layer", "") or "")
        layer = (
            requested_layer
            if requested_layer in context.layers_by_name
            else next(iter(context.layers_by_name), "")
        )
        if not layer:
            raise RuntimeError("目前沒有可用的 Score-CAM layer。")

        activation = context.layers_by_name[layer].detach()
        # Output logits are also valid spatial masks for Score-CAM. They have
        # only one channel per class, so report them as a distinct layer choice
        # rather than rejecting the tensor solely because it is model output.
        activation = activation.relu()
        channel_count = int(activation.size(1))
        start = max(0, min(int(params.get("_feature_start", 0)), channel_count))
        stop = max(start, min(int(params.get("_feature_stop", channel_count)), channel_count))
        if stop <= start:
            raise ValueError("Score-CAM feature 範圍不可為空。")

        fixed_target = None
        if params.get("_objective_id") == "predicted_target_mask":
            global_target = params.get("_fixed_target_mask")
            fixed_target = (
                torch.as_tensor(global_target, device=context.logits.device, dtype=torch.bool)
                if global_target is not None
                else context.logits.detach().argmax(dim=1) == context.target_class
            )
            if fixed_target.shape != context.logits[:, context.target_class].shape:
                raise ValueError("Score-CAM 固定目標 mask 與 tile 預測大小不一致。")
        target_present = fixed_target is None or bool(fixed_target.any())
        scores: list[torch.Tensor] = []
        valid_channels: list[bool] = []
        with torch.no_grad():
            for channel in range(start, stop):
                if not target_present:
                    scores.append(torch.zeros((), dtype=torch.float32))
                    valid_channels.append(False)
                    continue
                mask = F.interpolate(
                    activation[:, channel : channel + 1],
                    size=context.input_tensor.shape[2:],
                    mode="trilinear",
                    align_corners=False,
                )
                flat = mask.flatten(start_dim=2)
                minimum = flat.amin(dim=2, keepdim=True).view(mask.size(0), 1, 1, 1, 1)
                maximum = flat.amax(dim=2, keepdim=True).view(mask.size(0), 1, 1, 1, 1)
                mask_range = maximum - minimum
                constant = not bool((mask_range > 1e-12).all())
                if not bool(torch.isfinite(mask_range).all()) or (
                    constant and params.get("_objective_id") != "predicted_target_mask"
                ):
                    # A constant feature produces an empty normalized mask and
                    # cannot provide spatial evidence for this class.
                    scores.append(torch.zeros((), dtype=torch.float32))
                    valid_channels.append(False)
                    continue
                # Fixed-target benchmark includes zero masks for constant channels.
                mask = (mask - minimum) / mask_range.clamp_min(1e-8)
                masked_logits = context.model(context.input_tensor.detach() * mask)
                if fixed_target is not None:
                    target_logits = masked_logits[:, context.target_class]
                    score = target_score(masked_logits, context.target_class, fixed_target, protocol["reduction"])
                else:
                    score = context.objective(masked_logits, context.target_class)
                    score = score / masked_logits[0, 0].numel()
                finite_score = bool(torch.isfinite(score).all())
                scores.append(
                    score.detach().reshape(()).to("cpu")
                    if finite_score
                    else torch.zeros((), dtype=torch.float32)
                )
                valid_channels.append(finite_score)

        score_tensor = torch.stack(scores)
        cam = torch.zeros_like(activation[:, :1], dtype=torch.float32)
        weights_cpu = torch.zeros_like(score_tensor)
        if any(valid_channels):
            valid_scores = score_tensor.masked_fill(
                ~torch.tensor(valid_channels, dtype=torch.bool), float("-inf")
            )
            weights_cpu = torch.softmax(valid_scores, dim=0)
            weights = weights_cpu.to(activation.device)
            with torch.no_grad():
                for offset, channel in enumerate(range(start, stop)):
                    if valid_channels[offset]:
                        cam += activation[:, channel : channel + 1].float() * weights[offset]
                cam.clamp_min_(0)

        logger = params.get("_score_logger")
        if callable(logger):
            valid_count = sum(valid_channels)
            if valid_count:
                valid_scores = score_tensor[torch.tensor(valid_channels)]
                effective_count = 1.0 / float(weights_cpu.square().sum())
                logger(
                    f"Score-CAM tile {int(params.get('_tile_index', 0)) + 1}: "
                    f"layer={layer}, masked forwards={valid_count}, "
                    f"valid features={valid_count}/{stop - start}, "
                    f"score span={float(valid_scores.max() - valid_scores.min()):.4g}, "
                    f"max weight={float(weights_cpu.max()):.4g}, "
                    f"effective features={effective_count:.1f}"
                )
            else:
                logger(
                    f"Score-CAM tile {int(params.get('_tile_index', 0)) + 1}: "
                    f"layer={layer}, masked forwards=0, "
                    f"valid features=0/{stop - start}"
                )

        layer_payloads = {
            name: {"feature_count": int(value.size(1))}
            for name, value in context.layers_by_name.items()
        }
        if params.get("_retain_patch_payload"):
            layer_payloads[layer]["activation"] = activation.to("cpu")

        return {
            "method": self.id,
            "pred": context.logits.detach().to("cpu"),
            "selected_layer": layer,
            "feature_start": start,
            "feature_stop": stop,
            "target_present": target_present,
            "scores": score_tensor,
            "weights": weights_cpu,
            "masked_forward_count": sum(valid_channels),
            "valid_channels": torch.tensor(valid_channels, dtype=torch.bool),
            "cam": cam.to("cpu"),
            "layers": layer_payloads,
        }

    def _build_tile_cam(
        self,
        patch_payload: dict[str, object],
        selection: XaiLayerSelection,
        method_params: Mapping[str, object],
    ):
        import torch
        import torch.nn.functional as F

        selected_layer = str(patch_payload.get("selected_layer", ""))
        if selection.layer != selected_layer:
            raise ValueError(
                f"Score-CAM payload 是針對 layer '{selected_layer}' 計算，"
                f"無法改用 '{selection.layer}'。"
            )
        layers = patch_payload.get("layers")
        if not isinstance(layers, dict) or selection.layer not in layers:
            raise KeyError(f"layer '{selection.layer}' 不存在於 Score-CAM payload 中。")
        layer_payload = layers[selection.layer]
        if not isinstance(layer_payload, dict):
            raise TypeError("Score-CAM layer payload 格式錯誤。")
        activation = layer_payload.get("activation")
        stored_cam = patch_payload.get("cam")
        scores = patch_payload.get("scores")
        if not isinstance(scores, torch.Tensor) or not (
            isinstance(activation, torch.Tensor)
            or isinstance(stored_cam, torch.Tensor)
        ):
            raise TypeError("Score-CAM payload 缺少 CAM 或 activation/scores tensor。")

        stored_start = int(patch_payload.get("feature_start", 0))
        feature_count = activation.size(1) if isinstance(activation, torch.Tensor) else 0
        stored_stop = int(patch_payload.get("feature_stop", feature_count))
        start = max(stored_start, int(selection.n1))
        stop = min(stored_stop, int(selection.n2))
        if stop <= start:
            raise ValueError("指定的 Score-CAM feature 範圍未預先計算。")
        if isinstance(stored_cam, torch.Tensor):
            if start == stored_start and stop == stored_stop:
                return F.interpolate(
                    stored_cam.clamp_min(0),
                    size=selection.output_size,
                    mode="trilinear",
                    align_corners=False,
                )
            if not isinstance(activation, torch.Tensor):
                raise ValueError("Score-CAM feature 範圍已固定，請重新計算。")
        if patch_payload.get("target_present") is False:
            return torch.zeros((activation.size(0), 1, *selection.output_size))
        score_slice = scores[start - stored_start : stop - stored_start]
        valid_slice = patch_payload.get("valid_channels")
        if isinstance(valid_slice, torch.Tensor):
            valid_slice = valid_slice[start - stored_start : stop - stored_start].bool()
            if not bool(valid_slice.any()):
                return torch.zeros((activation.size(0), 1, *selection.output_size))
            score_slice = score_slice.masked_fill(~valid_slice, float("-inf"))
        weights = torch.softmax(score_slice.to(dtype=activation.dtype), dim=0).view(
            1, -1, 1, 1, 1
        )
        cam = torch.sum(activation[:, start:stop] * weights, dim=1, keepdim=True)
        return F.interpolate(
            cam.clamp_min(0),
            size=selection.output_size,
            mode="trilinear",
            align_corners=False,
        )
