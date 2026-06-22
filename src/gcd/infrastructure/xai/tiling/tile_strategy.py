from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class TileRegion:
    index: int
    origin: tuple[int, int, int]
    slices: tuple[slice, slice, slice]
    size: tuple[int, int, int]


@dataclass(frozen=True)
class TilePlan:
    regions: tuple[TileRegion, ...]
    input_shape: tuple[int, int, int]
    patch_size: tuple[int, int, int]
    strategy_id: str


class LegacyFourTileStrategy:
    id = "legacy_four_tile"

    def plan(
        self,
        *,
        input_shape: Sequence[int],
        patch_size: int | Sequence[int],
        stride: int | Sequence[int],
    ) -> TilePlan:
        shape = _shape3(input_shape)
        size = _patch3(patch_size)
        stride3 = _stride3(stride)
        x0, y0, z0 = (
            (shape[0] - (stride3[0] + size[0])) // 2,
            (shape[1] - (stride3[1] + size[1])) // 2,
            (shape[2] - size[2]) // 2,
        )
        offsets = [
            (0, 0, 0),
            (0, stride3[1], 0),
            (stride3[0], 0, 0),
            (stride3[0], stride3[1], 0),
        ]
        regions = []
        for index, (dx, dy, dz) in enumerate(offsets):
            origin = (int(x0 + dx), int(y0 + dy), int(z0 + dz))
            regions.append(_region(index, origin, size, shape))
        return TilePlan(
            regions=tuple(regions),
            input_shape=shape,
            patch_size=size,
            strategy_id=self.id,
        )


class SlidingWindowTileStrategy:
    id = "sliding_window"

    def plan(
        self,
        *,
        input_shape: Sequence[int],
        patch_size: int | Sequence[int],
        stride: int | Sequence[int],
    ) -> TilePlan:
        shape = _shape3(input_shape)
        size = tuple(min(patch, dim) for patch, dim in zip(_patch3(patch_size), shape))
        stride3 = _stride3(stride, minimum=1)
        starts = [_axis_starts(dim, patch, step) for dim, patch, step in zip(shape, size, stride3)]
        regions = []
        index = 0
        for x in starts[0]:
            for y in starts[1]:
                for z in starts[2]:
                    origin = (int(x), int(y), int(z))
                    regions.append(_region(index, origin, size, shape))
                    index += 1
        return TilePlan(
            regions=tuple(regions),
            input_shape=shape,
            patch_size=size,
            strategy_id=self.id,
        )


class TileStrategyResolver:
    def __init__(self) -> None:
        self._strategies = {
            LegacyFourTileStrategy.id: LegacyFourTileStrategy(),
            SlidingWindowTileStrategy.id: SlidingWindowTileStrategy(),
        }

    def resolve(self, method_params: Mapping[str, object] | None = None):
        strategy_id = str((method_params or {}).get("tile_strategy") or LegacyFourTileStrategy.id)
        if strategy_id == "legacy":
            strategy_id = LegacyFourTileStrategy.id
        if strategy_id not in self._strategies:
            raise ValueError(
                f"未知 tile_strategy '{strategy_id}'，可用策略: {', '.join(self._strategies)}"
            )
        return self._strategies[strategy_id]


def _shape3(value: Sequence[int]) -> tuple[int, int, int]:
    result = tuple(int(v) for v in value)
    if len(result) != 3:
        raise ValueError("tile input_shape 必須是 3D。")
    return result


def _patch3(value: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (int(value), int(value), int(value))
    result = tuple(int(v) for v in value)
    if len(result) != 3:
        raise ValueError("tile patch_size 必須是 int 或 3D tuple。")
    return tuple(max(1, v) for v in result)


def _stride3(value: int | Sequence[int], *, minimum: int = 0) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (max(minimum, int(value)),) * 3
    result = tuple(int(v) for v in value)
    if len(result) != 3:
        raise ValueError("tile stride 必須是 int 或 3D tuple。")
    return tuple(max(minimum, v) for v in result)


def _axis_starts(dim: int, patch: int, stride: int) -> list[int]:
    if patch >= dim:
        return [0]
    starts = list(range(0, dim - patch + 1, stride))
    last = dim - patch
    if starts[-1] != last:
        starts.append(last)
    return starts


def _region(
    index: int,
    origin: tuple[int, int, int],
    size: tuple[int, int, int],
    input_shape: tuple[int, int, int],
) -> TileRegion:
    starts = tuple(max(0, int(v)) for v in origin)
    stops = tuple(min(dim, start + patch) for start, patch, dim in zip(starts, size, input_shape))
    actual_size = tuple(max(0, stop - start) for start, stop in zip(starts, stops))
    return TileRegion(
        index=int(index),
        origin=starts,
        slices=tuple(slice(start, stop) for start, stop in zip(starts, stops)),
        size=actual_size,
    )
