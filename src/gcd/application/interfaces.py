from __future__ import annotations

from typing import Callable, Protocol


class VolumeRenderer(Protocol):
    def show_volumes(
        self,
        volumes: list[object],
        spacing: list[tuple[float, float, float]],
        metadata: list[dict[str, object] | None] | None = None,
        *,
        render_settings: list[dict[str, object]] | None = None,
        camera_policy: str = "preserve",
    ) -> None:
        ...

    def set_volume_transfer_functions(
        self,
        index: int,
        color_points: list[tuple[float, float, float, float]],
        opacity_points: list[tuple[float, float]],
        *,
        visible: bool | None = None,
        render: bool = True,
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
