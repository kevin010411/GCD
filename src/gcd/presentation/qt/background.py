from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import QThread, pyqtSignal


class _WorkerThread(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(object)

    def __init__(self, func: Callable[[], object]) -> None:
        super().__init__()
        self._func = func

    def run(self) -> None:
        try:
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
    ) -> None:
        worker = _WorkerThread(func)
        self._workers.append(worker)

        def _cleanup() -> None:
            if worker in self._workers:
                self._workers.remove(worker)
            worker.deleteLater()

        worker.succeeded.connect(on_success)
        worker.failed.connect(on_error)
        worker.finished.connect(_cleanup)
        worker.start()
