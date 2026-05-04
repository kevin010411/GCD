from __future__ import annotations

from PyQt6.QtWidgets import QApplication

from ..application.presenter import MainWindowPresenter
from ..application.services import TransferFunctionAppService, WorkflowService
from ..infrastructure.core_engine import GradCamEngine
from ..infrastructure.error_store import ErrorStore
from ..presentation.qt.background import QtBackgroundTaskRunner
from ..presentation.qt.styles import STYLESHEET
from ..presentation.qt.view import MainWindowView


def main(argv=None) -> None:
    app = QApplication(argv)
    app.setStyleSheet(STYLESHEET)

    view = MainWindowView()
    error_store = ErrorStore()
    workflow_service = WorkflowService(
        GradCamEngine("src/config/model/unet_3d.py", error_store=error_store)
    )
    presenter = MainWindowPresenter(
        view=view,
        workflow_service=workflow_service,
        transfer_service=TransferFunctionAppService(),
        task_runner=QtBackgroundTaskRunner(),
        error_store=error_store,
    )
    presenter.initialize()

    view.showMaximized()
    view.raise_()
    view.activateWindow()
    app.exec()
