from __future__ import annotations

from collections.abc import Callable
from inspect import signature

from PyQt6.QtCore import QThread, pyqtSignal


class _WorkerThread(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(object)
    progressed = pyqtSignal(object)

    def __init__(self, func: Callable[[], object]) -> None:
        super().__init__()
        self._func = func

    def run(self) -> None:
        try:
            if signature(self._func).parameters:
                self.succeeded.emit(self._func(self.progressed.emit))
            else:
                self.succeeded.emit(self._func())
        except Exception as exc:
            self.failed.emit(exc)


class QtBackgroundTaskRunner:
    def __init__(self) -> None:
        self._workers: list[_WorkerThread] = []

    def submit(
        self,
        func: Callable[[], object],
        on_success: Callable[[object], None],
        on_error: Callable[[Exception], None],
        on_progress: Callable[[object], None] | None = None,
    ) -> None:
        worker = _WorkerThread(func)
        self._workers.append(worker)

        def _cleanup() -> None:
            if worker in self._workers:
                self._workers.remove(worker)
            worker.deleteLater()

        worker.succeeded.connect(on_success)
        worker.failed.connect(on_error)
        if on_progress is not None:
            worker.progressed.connect(on_progress)
        worker.finished.connect(_cleanup)
        worker.start()
