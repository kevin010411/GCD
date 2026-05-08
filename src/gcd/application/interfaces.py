from __future__ import annotations

from typing import Callable, Protocol


class VolumeRenderer(Protocol):
    def show_volumes(
        self,
        volumes: list[object],
        spacing: list[tuple[float, float, float]],
        metadata: list[dict[str, object] | None] | None = None,
    ) -> None:
        ...

    def apply_transfer_function(
        self,
        color_points: list[tuple[float, float, float, float]],
        opacity_points: list[tuple[float, float]],
    ) -> None:
        ...


class BackgroundTaskRunner(Protocol):
    def submit(
        self,
        func: Callable[[], object],
        on_success: Callable[[object], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        ...
