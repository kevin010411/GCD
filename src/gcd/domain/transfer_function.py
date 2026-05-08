from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DataRange:
    min_value: float
    max_value: float

    def __post_init__(self) -> None:
        if self.max_value == self.min_value:
            object.__setattr__(self, "max_value", self.min_value + 1e-6)

    def clamp_position(self, position: float) -> float:
        return max(0.0, min(float(position), 1.0))

    def value_at(self, position: float) -> float:
        position = self.clamp_position(position)
        return self.min_value + position * (self.max_value - self.min_value)

    def position_for(self, value: float) -> float:
        return self.clamp_position(
            (float(value) - self.min_value) / (self.max_value - self.min_value)
        )

    @classmethod
    def from_data(
        cls,
        arrays: Iterable[object],
        method: str = "percentile",
        low_q: float = 1.0,
        high_q: float = 99.0,
    ) -> "DataRange":
        flat = []
        for array in arrays:
            if array is None:
                continue
            flat.append(np.ravel(np.asarray(array)))
        if not flat:
            return cls(0.0, 1.0)

        data = np.concatenate(flat)
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return cls(0.0, 1.0)

        if method == "minmax":
            return cls(float(np.nanmin(finite)), float(np.nanmax(finite)))
        return cls(
            float(np.nanpercentile(finite, low_q)),
            float(np.nanpercentile(finite, high_q)),
        )


@dataclass(frozen=True)
class ControlPoint:
    position: float
    color: str
    opacity: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", max(0.0, min(float(self.position), 1.0)))
        object.__setattr__(self, "opacity", max(0.0, min(float(self.opacity), 1.0)))


@dataclass(frozen=True)
class TransferFunction:
    control_points: tuple[ControlPoint, ...]

    def __post_init__(self) -> None:
        points = list(sorted(self.control_points, key=lambda point: point.position))
        if not points:
            raise ValueError("TransferFunction requires at least one control point.")
        if len(points) == 1:
            points[0] = ControlPoint(0.0, points[0].color, points[0].opacity)
        else:
            points[0] = ControlPoint(0.0, points[0].color, points[0].opacity)
            points[-1] = ControlPoint(1.0, points[-1].color, points[-1].opacity)
        object.__setattr__(self, "control_points", tuple(points))

    @classmethod
    def from_iterable(
        cls, control_points: Iterable[ControlPoint]
    ) -> "TransferFunction":
        return cls(tuple(control_points))

    @classmethod
    def overlay_preset(cls) -> "TransferFunction":
        return cls.from_iterable(
            [
                ControlPoint(0.000, "#9D5B2F", 0.0),
                ControlPoint(0.125, "#9D5B2F", 0.0),
                ControlPoint(0.1625, "#E19A4A", 0.65),
                ControlPoint(0.200, "#FFFFFF", 0.7),
                ControlPoint(0.325, "#FFFFFF", 0.8),
                ControlPoint(0.3725, "#FFFFFF", 0.8),
                ControlPoint(0.375, "#FFEFF4", 0.0),
                ControlPoint(0.500, "#CBCBCB", 0.0),
                ControlPoint(0.525, "#00008F", 0.0),
                ControlPoint(0.5375, "#00008F", 0.602),
                ControlPoint(0.550, "#00C3FF", 0.654),
                ControlPoint(0.700, "#FCFF03", 0.762),
                ControlPoint(0.825, "#FF7F00", 0.918),
                ControlPoint(1.000, "#FF2800", 1.0),
            ]
        )

    @classmethod
    def base_preset(cls) -> "TransferFunction":
        return cls.from_iterable(
            [
                ControlPoint(0, "#9D5B2F", 0.0),
                ControlPoint(0.3, "#9D5B2F", 0.0),
                ControlPoint(0.35, "#E19A4A", 0.65),
                ControlPoint(0.4, "#FFFFFF", 0.7),
                ControlPoint(0.5, "#FFFFFF", 0.8),
                ControlPoint(0.65, "#FFFFFF", 0.8),
                ControlPoint(0.875, "#FFEFF4", 0.0),
                ControlPoint(1, "#FFEFF4", 0.0),
            ]
        )

    @classmethod
    def heatmap_preset(cls) -> "TransferFunction":
        return cls.from_iterable(
            [
                ControlPoint(0, "#00008F", 0.0),
                ControlPoint(0.075, "#00008F", 0.602),
                ControlPoint(0.1, "#00C3FF", 0.654),
                ControlPoint(0.4, "#FCFF03", 0.762),
                ControlPoint(0.7, "#FF7F00", 0.918),
                ControlPoint(1, "#FF2800", 1.0),
            ]
        )

    def with_points(self, control_points: Iterable[ControlPoint]) -> "TransferFunction":
        return TransferFunction.from_iterable(control_points)

    def add_point(self, point: ControlPoint) -> "TransferFunction":
        return self.with_points([*self.control_points, point])

    def update_point(self, index: int, point: ControlPoint) -> "TransferFunction":
        points = list(self.control_points)
        points[index] = point
        return self.with_points(points)

    def remove_point(self, index: int) -> "TransferFunction":
        if len(self.control_points) == 1:
            return self
        points = list(self.control_points)
        del points[index]
        return self.with_points(points)

    def renderer_points(
        self, data_range: DataRange
    ) -> tuple[list[tuple[float, float, float, float]], list[tuple[float, float]]]:
        colors = []
        opacities = []
        for point in self.control_points:
            value = data_range.value_at(point.position)
            color = point.color.lstrip("#")
            r = int(color[0:2], 16) / 255.0
            g = int(color[2:4], 16) / 255.0
            b = int(color[4:6], 16) / 255.0
            colors.append((value, r, g, b))
            opacities.append((value, point.opacity))
        return colors, opacities
