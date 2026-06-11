from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from .base import XaiMethod
from .gradcam import GradCamMethod
from .gradcam_test import GradCAMTestMethod
from .perturb_occlusion import PerturbationOcclusionMethod
from .saliency_map import SaliencyMapMethod
from .xrescam import XResCamMethod

if TYPE_CHECKING:
    import torch


class XaiMethodRegistry:
    def __init__(self, methods: Iterable[XaiMethod]) -> None:
        self._methods = {method.id: method for method in methods}

    @classmethod
    def default(
        cls, objective: Callable[[torch.Tensor, int], torch.Tensor] | None = None
    ) -> XaiMethodRegistry:
        objective = objective or (lambda logits, target_class: logits[0, target_class].sum())
        return cls(
            (
                GradCamMethod(objective),
                XResCamMethod(objective),
                GradCAMTestMethod(),
                SaliencyMapMethod(objective),
                PerturbationOcclusionMethod(),
            )
        )

    @property
    def methods_by_id(self) -> dict[str, XaiMethod]:
        return dict(self._methods)

    def resolve(self, method_id: str | None, fallback: str = "gradcam") -> XaiMethod:
        requested = (method_id or fallback).strip().lower()
        return self._methods.get(requested) or self._methods[fallback]

    def available_families(self) -> list[dict[str, object]]:
        labels = {
            "gradient": ("Gradient XAI", "Gradient"),
            "perturbation": ("Perturbation XAI", "Perturb"),
        }
        seen: list[str] = []
        for method in self._methods.values():
            if method.family not in seen:
                seen.append(method.family)
        return [
            {
                "id": family,
                "title": labels.get(family, (family.title(), family.title()))[0],
                "button_label": labels.get(family, (family.title(), family.title()))[1],
            }
            for family in seen
        ]

    def available_methods(self, family: str | None = None) -> list[dict[str, object]]:
        methods = self._methods.values()
        if family is not None:
            methods = [method for method in methods if method.family == family]
        return [method.definition().to_option() for method in methods]
