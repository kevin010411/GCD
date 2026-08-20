from __future__ import annotations

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QAction, QColor, QLinearGradient, QPainter
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .plugins import DEFAULT_PLUGIN_DEFINITIONS, PluginDefinition
from .styles import CLASSIC_STYLESHEET, DARK_STYLESHEET
from .workspace import WorkspaceHost


class _ScrollFade(QWidget):
    def __init__(self, edge: str, parent=None) -> None:
        super().__init__(parent)
        self.edge = edge
        self.base_color = QColor("#151F2F")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFixedWidth(26)

    def set_base_color(self, color: str) -> None:
        self.base_color = QColor(color)
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        gradient = QLinearGradient()
        if self.edge == "left":
            gradient = QLinearGradient(self.width(), 0, 0, 0)
        else:
            gradient = QLinearGradient(0, 0, self.width(), 0)
        base = QColor(self.base_color)
        transparent = QColor(base)
        transparent.setAlpha(0)
        gradient.setColorAt(0.0, base)
        gradient.setColorAt(1.0, transparent)
        painter.fillRect(self.rect(), gradient)


class PluginTabStrip(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pluginSwitchStrip")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.scroll = QScrollArea(self)
        self.scroll.setObjectName("pluginSwitchScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.scroll.setFixedHeight(42)
        self.scroll.viewport().installEventFilter(self)

        self.content = QWidget()
        self.content.setObjectName("pluginSwitchContent")
        self.row = QFrame()
        self.row.setObjectName("pluginSwitchRow")
        self.row_layout = QHBoxLayout(self.row)
        self.row_layout.setContentsMargins(0, 0, 0, 0)
        self.row_layout.setSpacing(8)
        self.row_layout.addStretch(1)

        content_layout = QHBoxLayout(self.content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        content_layout.addWidget(self.row, 0, Qt.AlignmentFlag.AlignLeft)
        content_layout.addStretch(1)
        self.scroll.setWidget(self.content)

        self.left_fade = _ScrollFade("left", self)
        self.right_fade = _ScrollFade("right", self)

        layout.addWidget(self.scroll, 1)
        self.left_fade.raise_()
        self.right_fade.raise_()

        scrollbar = self.scroll.horizontalScrollBar()
        scrollbar.valueChanged.connect(self._update_fades)
        scrollbar.rangeChanged.connect(self._update_fades)
        self._update_fades()

    def add_button(self, button: QPushButton) -> None:
        insert_index = max(0, self.row_layout.count() - 1)
        self.row_layout.insertWidget(insert_index, button)
        self._update_fades()

    def center_button(self, button: QPushButton) -> None:
        scrollbar = self.scroll.horizontalScrollBar()
        content_pos = button.mapTo(self.content, button.rect().topLeft())
        button_center = content_pos.x() + button.width() // 2
        viewport_half = self.scroll.viewport().width() // 2
        target = button_center - viewport_half
        scrollbar.setValue(max(scrollbar.minimum(), min(target, scrollbar.maximum())))
        self._update_fades()

    def scroll_to_start(self) -> None:
        scrollbar = self.scroll.horizontalScrollBar()
        scrollbar.setValue(scrollbar.minimum())
        self._update_fades()

    def set_fade_color(self, color: str) -> None:
        self.left_fade.set_base_color(color)
        self.right_fade.set_base_color(color)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        height = self.scroll.height()
        self.left_fade.setGeometry(0, 0, self.left_fade.width(), height)
        self.right_fade.setGeometry(
            max(0, self.width() - self.right_fade.width()),
            0,
            self.right_fade.width(),
            height,
        )

    def eventFilter(self, source, event) -> bool:
        if source is self.scroll.viewport() and event.type() == QEvent.Type.Wheel:
            delta = event.angleDelta().y()
            if delta:
                scrollbar = self.scroll.horizontalScrollBar()
                step = max(40, self.width() // 5)
                direction = -1 if delta > 0 else 1
                scrollbar.setValue(scrollbar.value() + direction * step)
                return True
        return super().eventFilter(source, event)

    def _update_fades(self, *_args) -> None:
        scrollbar = self.scroll.horizontalScrollBar()
        maximum = scrollbar.maximum()
        value = scrollbar.value()
        self.left_fade.setVisible(value > 0)
        self.right_fade.setVisible(maximum > 0 and value < maximum)


class MainWindowView(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Grad-CAM Plugin Workspace")
        self.plugin_definitions: tuple[PluginDefinition, ...] = DEFAULT_PLUGIN_DEFINITIONS
        self.plugin_titles = {
            definition.plugin_id: definition.title
            for definition in self.plugin_definitions
        }
        self._classic_theme_enabled = False
        self._method_options_by_id: dict[str, dict[str, object]] = {}
        self._xai_method_options_by_family: dict[str, dict[str, dict[str, object]]] = {}
        self._gradcam_feature_size = 0

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(18, 18, 18, 18)
        self.main_layout.setSpacing(14)

        self._build_global_toolbar()

        body = QHBoxLayout()
        body.setSpacing(14)
        self.main_layout.addLayout(body, 1)

        self.workspace = WorkspaceHost()
        self.renderer = self.workspace
        body.addWidget(self.workspace, 1)

        self.inspector_frame = QFrame()
        self.inspector_frame.setObjectName("pluginStack")
        inspector_layout = QVBoxLayout(self.inspector_frame)
        inspector_layout.setContentsMargins(0, 0, 0, 0)
        inspector_layout.setSpacing(0)
        self.inspector_header = QFrame()
        self.inspector_header.setObjectName("inspectorHeader")
        header_layout = QVBoxLayout(self.inspector_header)
        header_layout.setContentsMargins(16, 14, 16, 14)
        header_layout.setSpacing(8)
        self.inspector_kicker = QLabel("Plugin")
        self.inspector_kicker.setObjectName("inspectorKicker")
        header_layout.addWidget(self.inspector_kicker)
        self.plugin_switch_strip = PluginTabStrip(self)
        self.plugin_button_group = QButtonGroup(self)
        self.plugin_button_group.setExclusive(True)
        self.plugin_buttons: dict[str, QPushButton] = {}
        for definition in self.plugin_definitions:
            button = QPushButton(definition.button_label)
            button.setObjectName("pluginTabButton")
            button.setCheckable(True)
            button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            self.plugin_button_group.addButton(button)
            self.plugin_switch_strip.add_button(button)
            self.plugin_buttons[definition.plugin_id] = button
            setattr(self, f"{definition.plugin_id}_plugin_button", button)
        self.transfer_plugin_button = self.plugin_buttons["data"]
        header_layout.addWidget(self.plugin_switch_strip)
        inspector_layout.addWidget(self.inspector_header)
        self.plugin_stack = QStackedWidget()
        self.plugin_stack.setObjectName("pluginStack")
        inspector_layout.addWidget(self.plugin_stack)
        self.inspector_frame.setMinimumWidth(320)
        self.inspector_frame.setMaximumWidth(420)
        body.addWidget(self.inspector_frame)

        self._build_registered_plugins()
        self.set_active_plugin("gradcam")
        self.plugin_switch_strip.scroll_to_start()

    def _build_global_toolbar(self) -> None:
        toolbar = QFrame()
        toolbar.setObjectName("globalToolbar")
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)

        title_layout = QVBoxLayout()
        title = QLabel("Plugin Workbench")
        title.setObjectName("appTitle")
        subtitle = QLabel("Shared 3D viewport with flexible 2D slice tiles")
        subtitle.setObjectName("appSubtitle")
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        layout.addLayout(title_layout, 1)

        layout.addWidget(QLabel("Model"))
        self.model_combo = QComboBox(self)
        self.model_combo.setObjectName("modelCombo")
        self.model_combo.setMinimumWidth(180)
        if self.model_combo.view() is not None:
            self.model_combo.view().setObjectName("softComboPopup")
        layout.addWidget(self.model_combo)

        self.workbench_button = QToolButton()
        self.workbench_button.setObjectName("layoutButton")
        self.workbench_button.setText("Workbench")
        self.workbench_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.workbench_menu = QMenu(self.workbench_button)
        self.workbench_menu.setObjectName("softMenuPopup")
        self.workbench_button.setMenu(self.workbench_menu)

        self.layout_action_focus = QAction("Focus 3D", self)
        self.layout_action_triple = QAction("3D + Triple Slice", self)
        self.layout_action_3d_only = QAction("3D Only", self)
        self.layout_action_compare = QAction("Compare", self)
        self.workbench_menu.clear()
        self.workbench_menu.addSection("Layouts")
        self.workbench_menu.addAction(self.layout_action_focus)
        self.workbench_menu.addAction(self.layout_action_triple)
        self.workbench_menu.addAction(self.layout_action_3d_only)
        self.workbench_menu.addAction(self.layout_action_compare)
        layout.addWidget(self.workbench_button)

        self.inspector_toggle_button = QPushButton("Inspector")
        self.inspector_toggle_button.setCheckable(True)
        self.inspector_toggle_button.setChecked(True)
        layout.addWidget(self.inspector_toggle_button)

        self.open_file_button = QPushButton("Open Volume")
        self.save_screenshot_button = QPushButton("Screenshot")
        self.record_video_button = QPushButton("Record")
        layout.addWidget(self.open_file_button)
        layout.addWidget(self.save_screenshot_button)
        layout.addWidget(self.record_video_button)

        self.theme_toggle_button = QToolButton()
        self.theme_toggle_button.setObjectName("themeToggleButton")
        self.theme_toggle_button.setText("☾")
        self.theme_toggle_button.setToolTip("Switch to classic style")
        self.theme_toggle_button.clicked.connect(self.toggle_style_theme)
        layout.addWidget(self.theme_toggle_button)

        self.main_layout.addWidget(toolbar)

    def toggle_style_theme(self, _checked: bool = False) -> None:
        self._classic_theme_enabled = not self._classic_theme_enabled
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(
                CLASSIC_STYLESHEET
                if self._classic_theme_enabled
                else DARK_STYLESHEET
            )
        self.theme_toggle_button.setText("☀" if self._classic_theme_enabled else "☾")
        self.theme_toggle_button.setToolTip(
            "Switch to dark style"
            if self._classic_theme_enabled
            else "Switch to classic style"
        )
        self.plugin_switch_strip.set_fade_color(
            "#F0F0F0" if self._classic_theme_enabled else "#151F2F"
        )

    def _build_registered_plugins(self) -> None:
        self.plugin_panels: dict[str, QWidget] = {}
        for definition in self.plugin_definitions:
            panel = definition.panel_factory(self)
            self.plugin_panels[definition.plugin_id] = panel
            setattr(self, f"{definition.plugin_id}_plugin_panel", panel)
            self._add_plugin_tab(definition.plugin_id, panel)
        self._expose_registered_plugin_controls()

    def _expose_registered_plugin_controls(self) -> None:
        self.gradcam_plugin_panel = self.plugin_panels["gradcam"]
        self.xai_family_panels = {
            "gradient": self.gradcam_plugin_panel,
            "perturbation": self.plugin_panels["perturbation"],
        }
        self.class_spinbox = self.gradcam_plugin_panel.class_spinbox
        self.gradcam_dataset_combo = self.gradcam_plugin_panel.dataset_combo
        self.layer_combo = self.gradcam_plugin_panel.layer_combo
        self.layer_feature_group = self.gradcam_plugin_panel.layer_feature_group
        self.method_combo = self.gradcam_plugin_panel.method_combo
        self.objective_combo = self.gradcam_plugin_panel.objective_combo
        self.feature_widget = self.gradcam_plugin_panel.feature_widget
        self.gradcam_run_button = self.gradcam_plugin_panel.run_button

        self.perturbation_plugin_panel = self.plugin_panels["perturbation"]
        self.perturbation_dataset_combo = self.perturbation_plugin_panel.dataset_combo
        self.perturbation_class_spinbox = self.perturbation_plugin_panel.class_spinbox
        self.perturbation_method_combo = self.perturbation_plugin_panel.method_combo
        self.perturbation_run_button = self.perturbation_plugin_panel.run_button
        self.perturbation_preview_pause_button = (
            self.perturbation_plugin_panel.preview_pause_button
        )
        self.organ_occlusion_controls = self.perturbation_plugin_panel.organ_controls
        self.organ_total_button = self.organ_occlusion_controls.total_button
        self.organ_rerun_button = self.organ_occlusion_controls.rerun_button

        self.data_plugin_panel = self.plugin_panels["data"]
        self.transfer_plugin_panel = self.data_plugin_panel
        self.volume_list = self.data_plugin_panel.volume_list
        self.delete_volume_button = self.data_plugin_panel.delete_button
        self.save_volume_button = self.data_plugin_panel.save_button
        self.reorder_hint_label = self.data_plugin_panel.reorder_hint
        self.overlay_status_label = self.data_plugin_panel.overlay_status
        self.transfer_editor = self.data_plugin_panel.transfer_editor

        self.roi_plugin_panel = self.plugin_panels["roi"]
        self.roi_mode_combo = self.roi_plugin_panel.mode_combo
        self.roi_point_size_slider = self.roi_plugin_panel.point_size_slider
        self.roi_point_size_spinbox = self.roi_plugin_panel.point_size_spinbox
        self.roi_box_combo = self.roi_plugin_panel.roi_box_combo
        self.roi_import_button = self.roi_plugin_panel.import_button
        self.roi_export_button = self.roi_plugin_panel.export_button
        self.roi_delete_selected_button = self.roi_plugin_panel.delete_selected_button
        self.roi_clear_all_button = self.roi_plugin_panel.clear_all_button
        self.roi_annotation_list = self.roi_plugin_panel.annotation_list

        self.camera_plugin_panel = self.plugin_panels["camera"]
        self.speed_label = self.camera_plugin_panel.speed_label
        self.speed_slider = self.camera_plugin_panel.speed_slider
        self.start_button = self.camera_plugin_panel.start_button
        self.stop_button = self.camera_plugin_panel.stop_button
        self.replace_camera_button = self.camera_plugin_panel.reset_camera_button
        self.import_camera_button = self.camera_plugin_panel.import_camera_button
        self.export_camera_button = self.camera_plugin_panel.export_camera_button

    def _add_plugin_tab(self, plugin_id: str, widget: QWidget) -> None:
        index = self.plugin_stack.addWidget(widget)
        setattr(self, f"{plugin_id}_plugin_index", index)

    def set_active_plugin(self, plugin_id: str) -> None:
        if plugin_id == "transfer":
            plugin_id = "data"
        index = getattr(self, f"{plugin_id}_plugin_index")
        self.plugin_stack.setCurrentIndex(index)
        for current_id, button in self.plugin_buttons.items():
            button.setChecked(current_id == plugin_id)
        active_button = self.plugin_buttons[plugin_id]
        self.plugin_switch_strip.center_button(active_button)
        definition = next(
            item for item in self.plugin_definitions if item.plugin_id == plugin_id
        )
        if definition.workspace_mode == "roi":
            self.workspace.set_workspace_mode("roi")
        elif definition.workspace_mode == "standard":
            self.workspace.set_workspace_mode("standard")
        reorder_enabled = plugin_id == "data" and self.workspace.mode.value == "roi"
        self.volume_list.set_reorder_enabled(reorder_enabled)
        self.reorder_hint_label.setText(
            "Drag to reorder ROI drawing priority."
            if reorder_enabled
            else "Reorder is disabled in standard mode."
        )
        self.inspector_frame.show()
        self.inspector_toggle_button.setChecked(True)

    def toggle_inspector(self, visible: bool) -> None:
        self.inspector_frame.setVisible(visible)
        self.inspector_toggle_button.setChecked(visible)

    def set_model_options(self, options: list[dict[str, str]]) -> None:
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for option in options:
            self.model_combo.addItem(option["name"], option["path"])
        self.model_combo.blockSignals(False)

    def set_layer_options(self, layer_names: list[str], selected: str) -> None:
        if "xai_family_panels" not in self.__dict__:
            self.layer_combo.blockSignals(True)
            self.layer_combo.clear()
            self.layer_combo.addItems(layer_names)
            self.layer_combo.setCurrentText(selected)
            self.layer_combo.blockSignals(False)
            self.set_gradcam_layer_controls_enabled(
                self.selected_method_uses_layer_controls()
            )
            return
        self.set_xai_layer_options("gradient", layer_names, selected, self._gradcam_feature_size)

    def set_method_options(
        self, options: list[dict[str, object]], selected: str | None
    ) -> None:
        self.set_xai_method_options("gradient", options, selected)

    def set_xai_method_options(
        self, family_id: str, options: list[dict[str, object]], selected: str | None
    ) -> None:
        if "xai_family_panels" not in self.__dict__:
            self.method_combo.blockSignals(True)
            self.method_combo.clear()
            self._method_options_by_id = {
                str(option["id"]): dict(option) for option in options
            }
            for option in options:
                self.method_combo.addItem(str(option["name"]), str(option["id"]))
            if selected:
                index = self.method_combo.findData(selected)
                if index >= 0:
                    self.method_combo.setCurrentIndex(index)
            self.method_combo.blockSignals(False)
            self.set_gradcam_layer_controls_enabled(
                self.selected_method_uses_layer_controls()
            )
            return
        panel = self.xai_family_panels[family_id]
        self._method_options_by_id = {str(option["id"]): dict(option) for option in options}
        self._xai_method_options_by_family[family_id] = {
            str(option["id"]): dict(option) for option in options
        }
        panel.set_method_options(options, selected)

    def set_objective_options(
        self, options: list[dict[str, object]], selected: str | None
    ) -> None:
        for panel in self.xai_family_panels.values():
            panel.set_objective_options(options, selected)

    def set_xai_objective_options(
        self, family_id: str, options: list[dict[str, object]], selected: str | None
    ) -> None:
        self.xai_family_panels[family_id].set_objective_options(options, selected)

    def set_gradcam_dataset_options(
        self, options: list[dict[str, str]], selected: str | None
    ) -> None:
        self.set_xai_dataset_options("gradient", options, selected)

    def set_perturbation_dataset_options(
        self, options: list[dict[str, str]], selected: str | None
    ) -> None:
        self.set_xai_dataset_options("perturbation", options, selected)

    def set_xai_dataset_options(
        self, family_id: str, options: list[dict[str, str]], selected: str | None
    ) -> None:
        self.xai_family_panels[family_id].set_dataset_options(options, selected)

    def set_perturbation_answer_data_options(
        self, options: list[dict[str, str]], selected: str | None
    ) -> None:
        self.xai_family_panels["perturbation"].set_answer_data_options(
            options, selected
        )

    def set_perturbation_progress_running(self, running: bool) -> None:
        self.xai_family_panels["perturbation"].set_progress_running(running)

    def set_perturbation_progress(self, current: int, total: int) -> None:
        self.xai_family_panels["perturbation"].set_progress_value(current, total)

    def set_perturbation_preview_running(self, running: bool) -> None:
        self.xai_family_panels["perturbation"].set_preview_running(running)

    def set_organ_masks(self, records: list[dict[str, object]]) -> None:
        self.perturbation_plugin_panel.set_organ_masks(records)

    def set_organ_status(self, text: str, *, error: bool = False) -> None:
        self.organ_occlusion_controls.set_status(text, error=error)

    def set_organ_running(self, running: bool) -> None:
        self.perturbation_plugin_panel.set_organ_running(running)

    def set_organ_metrics(self, metrics: dict[str, float]) -> None:
        self.organ_occlusion_controls.set_metrics(metrics)

    def selected_organ_interventions(self) -> list[dict[str, object]]:
        return self.organ_occlusion_controls.selected_interventions()

    def set_perturbation_method_options(
        self, options: list[dict[str, str]], selected: str | None
    ) -> None:
        self.set_xai_method_options("perturbation", options, selected)

    def set_feature_size(self, size: int) -> None:
        self._gradcam_feature_size = max(0, int(size))
        if "xai_family_panels" not in self.__dict__:
            self.feature_widget.set_size(size)
            self.set_gradcam_layer_controls_enabled(
                self.selected_method_uses_layer_controls()
            )
            return
        self.set_xai_layer_options(
            "gradient",
            [self.layer_combo.itemText(index) for index in range(self.layer_combo.count())],
            self.layer_combo.currentText(),
            self._gradcam_feature_size,
        )

    def set_xai_layer_options(
        self,
        family_id: str,
        layer_names: list[str],
        selected: str,
        feature_size: int,
    ) -> None:
        self.xai_family_panels[family_id].set_layer_options(
            layer_names, selected, feature_size
        )

    def set_gradcam_layer_controls_enabled(self, enabled: bool) -> None:
        if "gradcam_plugin_panel" not in self.__dict__:
            can_select_layer = bool(enabled) and self.layer_combo.count() > 0
            can_adjust_feature = can_select_layer and self._gradcam_feature_size > 0
            self.layer_feature_group.setEnabled(can_select_layer)
            self.layer_combo.setEnabled(can_select_layer)
            self.feature_widget.setEnabled(can_adjust_feature)
            return
        self.gradcam_plugin_panel._sync_capability_controls()

    def set_rotation_speed_label(self, speed: float) -> None:
        self.speed_label.setText(f"Rotation Speed: {speed:.1f}")

    def set_rotation_running(self, is_running: bool) -> None:
        self.start_button.setEnabled(not is_running)
        self.stop_button.setEnabled(is_running)

    def set_overlay_status_message(self, message: str) -> None:
        self.overlay_status_label.setText(message)

    def set_data_controls_enabled(self, enabled: bool) -> None:
        if hasattr(self.data_plugin_panel, "set_data_controls_enabled"):
            self.data_plugin_panel.set_data_controls_enabled(enabled)

    def set_camera_controls_enabled(self, enabled: bool) -> None:
        if hasattr(self.camera_plugin_panel, "set_camera_controls_enabled"):
            self.camera_plugin_panel.set_camera_controls_enabled(enabled)
        if hasattr(self.workspace, "set_camera_interaction_enabled"):
            self.workspace.set_camera_interaction_enabled(enabled)

    def selected_model_path(self) -> str | None:
        return self.model_combo.currentData()

    def selected_layer(self) -> str:
        return self.layer_combo.currentText()

    def selected_class(self) -> int:
        return self.class_spinbox.value()

    def selected_method(self) -> str:
        current = self.method_combo.currentData()
        return str(current or "gradcam")

    def selected_objective(self) -> str:
        current = self.objective_combo.currentData()
        return str(current or "predicted_target_mask")

    def selected_method_uses_layer_controls(self) -> bool:
        if "gradcam_plugin_panel" not in self.__dict__:
            option = self._method_options_by_id.get(self.selected_method())
            if option is None:
                return True
            return bool(option.get("uses_layer_controls", True))
        return self.gradcam_plugin_panel.selected_method_uses_layer_controls()

    def selected_gradcam_dataset(self) -> str:
        current = self.gradcam_dataset_combo.currentData()
        return str(current or "")

    def selected_perturbation_dataset(self) -> str:
        current = self.perturbation_dataset_combo.currentData()
        return str(current or "")

    def selected_perturbation_method(self) -> str:
        return self.selected_xai_method("perturbation") or "perturb_occlusion"

    def feature_range(self) -> tuple[int, int]:
        return self.feature_widget.get_range()

    def selected_xai_dataset(self, family_id: str) -> str:
        return self.xai_family_panels[family_id].selected_dataset()

    def selected_perturbation_answer_data(self) -> str:
        return self.xai_family_panels["perturbation"].selected_answer_data()

    def selected_perturbation_preview_enabled(self) -> bool:
        return self.xai_family_panels["perturbation"].preview_enabled()

    def selected_xai_method(self, family_id: str) -> str:
        return self.xai_family_panels[family_id].selected_method()

    def selected_xai_objective(self, family_id: str) -> str:
        return self.xai_family_panels[family_id].selected_objective()

    def selected_xai_class(self, family_id: str) -> int:
        return self.xai_family_panels[family_id].selected_class()

    def selected_xai_layer(self, family_id: str) -> str:
        return self.xai_family_panels[family_id].selected_layer()

    def selected_xai_feature_range(self, family_id: str) -> tuple[int, int]:
        return self.xai_family_panels[family_id].feature_range()

    def selected_xai_method_params(self, family_id: str) -> dict[str, object]:
        return self.xai_family_panels[family_id].selected_method_params()

    def selected_xai_method_uses_layer_controls(self, family_id: str) -> bool:
        return self.xai_family_panels[family_id].selected_method_uses_layer_controls()

    def selected_xai_method_uses_objective(self, family_id: str) -> bool:
        return self.xai_family_panels[family_id].selected_method_uses_objective()

    def choose_input_file(self) -> str:
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open NIfTI File", "", "NIfTI Files (*.nii.gz *.nii)"
        )
        return file_name

    def choose_screenshot_file(self) -> str:
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", "screenshot", "PNG Files (*.png)"
        )
        return file_name[:-4] if file_name.lower().endswith(".png") else file_name

    def choose_volume_save_file(self, default_name: str = "volume.nii.gz") -> str:
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Volume", default_name, "NIfTI Files (*.nii.gz *.nii)"
        )
        return file_name

    def choose_video_file(self) -> str:
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Video", "rotation_video.mp4", "MP4 Files (*.mp4)"
        )
        return file_name

    def choose_camera_import_file(self) -> str:
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Import Camera", "", "JSON Files (*.json)"
        )
        return file_name

    def choose_camera_export_file(self) -> str:
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Export Camera", "camera.json", "JSON Files (*.json)"
        )
        return file_name

    def choose_annotation_import_file(self) -> str:
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Import Annotations", "", "JSON Files (*.json)"
        )
        return file_name

    def choose_annotation_export_file(self) -> str:
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Export Annotations", "annotations.json", "JSON Files (*.json)"
        )
        return file_name

    def closeEvent(self, event) -> None:
        try:
            self.workspace.shutdown()
        finally:
            super().closeEvent(event)
