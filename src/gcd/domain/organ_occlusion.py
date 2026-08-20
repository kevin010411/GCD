from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OrganMaskRecord:
    id: str
    display_name: str
    source_labels: tuple[str, ...]
    mask: Any
    affine: Any
    color: str = "#F59E0B"
    group: str = "advanced"


@dataclass(frozen=True)
class OrganIntervention:
    organ_id: str
    enabled: bool = False
    visible: bool = True
    mode: str = "local_mean"
    fill_hu: float = 0.0


@dataclass(frozen=True)
class OrganOcclusionSpec:
    interventions: tuple[OrganIntervention, ...]
    feather_mm: float = 2.0
    blur_sigma_mm: float = 3.0
    local_mean_shell_mm: float = 5.0

    def enabled(self) -> tuple[OrganIntervention, ...]:
        return tuple(item for item in self.interventions if item.enabled)


@dataclass(frozen=True)
class OrganOcclusionResult:
    occluded_ct: Any
    original_prediction: Any
    perturbed_prediction: Any
    difference: Any
    modified_mask: Any
    metrics: dict[str, float]
    spec: OrganOcclusionSpec
    metadata: dict[str, object] = field(default_factory=dict)

