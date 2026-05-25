# Plugin Panels

This folder contains the right-side plugin panels used by the Qt workbench UI.

Current plugins:

- `gradcam.py`
  - `GradCamPluginPanel`
- `roi_annotation.py`
  - `RoiAnnotationPluginPanel`
- `transfer_volume.py`
  - `TransferVolumePluginPanel` (Data plugin)

These files only define the panel UI and the controls exposed by each plugin.
Behavior is still connected in:

- `src/gcd/application/presenter.py`

The plugin switch area is assembled from the registry in:

- `src/gcd/presentation/qt/view.py`
- `src/gcd/presentation/qt/plugins/registry.py`

## Current structure

Each plugin panel should:

- inherit from `PluginPanel` in `base.py`
- build its own Qt widgets
- expose the widgets as instance attributes
- avoid embedding workflow or renderer business logic directly

## How to create a new plugin

1. Add a new file in this folder.

Example:

- `roi_volume.py`
- class name: `RoiVolumePluginPanel`

2. In that file, subclass `PluginPanel`.

Minimal pattern:

```python
from .base import PluginPanel


class RoiVolumePluginPanel(PluginPanel):
    def __init__(self, parent=None) -> None:
        super().__init__(
            "ROI Volume",
            "Control ROI-driven rendering and visibility.",
            parent,
        )
        # add widgets to self.content_layout
```

3. Expose all controls that the presenter needs as attributes on the panel.

Examples:

- `self.threshold_slider`
- `self.roi_combo`
- `self.apply_button`

## How to add the plugin into the switch area

1. Import the panel in `src/gcd/presentation/qt/view.py`.

2. Create the panel instance and add it to the plugin stack.

Pattern:

```python
self.roi_plugin_panel = RoiVolumePluginPanel(self)
self._add_plugin_tab("roi", self.roi_plugin_panel)
```

3. Create a switch button in the inspector header.

Pattern:

```python
self.roi_plugin_button = QPushButton("ROI")
self.roi_plugin_button.setObjectName("pluginTabButton")
self.roi_plugin_button.setCheckable(True)
self.plugin_button_group.addButton(self.roi_plugin_button)
switch_layout.addWidget(self.roi_plugin_button, 1)
```

4. Extend `self.plugin_titles` in `MainWindowView`.

Example:

```python
self.plugin_titles["roi"] = "ROI Volume"
```

5. Update `set_active_plugin()` in `MainWindowView` so the new button reflects the active plugin.

6. Connect the new switch button in `src/gcd/application/presenter.py`.

Pattern:

```python
self.view.roi_plugin_button.clicked.connect(
    lambda: self.view.set_active_plugin("roi")
)
```

7. Connect the plugin's internal controls in `presenter.py`.

This is where the plugin becomes functional.

## Recommended rules

- Keep plugin UI here.
- Keep plugin behavior in the presenter or lower application layers.
- Keep layout switching logic out of the plugin panel itself unless the plugin intentionally requests a workspace layout.
- Reuse the shared `ViewerWorkspace` instead of building a separate renderer per plugin.
