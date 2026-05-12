STYLESHEET = """
    QWidget {
        color: #E8ECF3;
        font-size: 13px;
    }

    QMainWindow, QWidget#centralwidget {
        background: #0C1118;
    }

    QFrame#globalToolbar, QFrame#workspaceToolbar {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 #111B29, stop:1 #162333);
        border: 1px solid #253248;
        border-radius: 18px;
    }

    QLabel#appTitle {
        font-size: 22px;
        font-weight: 700;
        color: #F2F6FC;
    }

    QLabel#appSubtitle {
        color: #9FB0C5;
        font-size: 12px;
    }

    QLabel#panelTitle {
        font-size: 20px;
        font-weight: 700;
        color: #F2F6FC;
    }

    QLabel#panelDescription, QLabel#workspaceHint {
        color: #9FB0C5;
    }

    QLabel#statusPill {
        padding: 8px 12px;
        border-radius: 12px;
        background: #142032;
        border: 1px solid #2A3850;
        color: #CFE3FF;
        font-weight: 600;
    }

    QListWidget#pluginRail {
        background: #111723;
        border: 1px solid #202C3F;
        border-radius: 20px;
        padding: 10px;
        outline: none;
    }

    QListWidget#pluginRail::item {
        background: #151E2E;
        border: 1px solid #243046;
        border-radius: 16px;
        margin: 6px;
        padding: 18px 10px;
        min-height: 42px;
        color: #B8C8DC;
        font-weight: 600;
    }

    QListWidget#pluginRail::item:selected {
        background: #1F3552;
        border: 1px solid #4C7EB8;
        color: #F4F8FF;
    }

    QFrame#pluginStack, QStackedWidget#pluginStack {
        background: #111723;
        border: 1px solid #202C3F;
        border-radius: 20px;
    }

    QFrame#inspectorHeader {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 #151F2F, stop:1 #1A283B);
        border-bottom: 1px solid #243754;
        border-top-left-radius: 20px;
        border-top-right-radius: 20px;
    }

    QLabel#inspectorKicker {
        color: #8FA6C5;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    QWidget#pluginSwitchStrip,
    QWidget#pluginSwitchContent,
    QFrame#pluginSwitchRow {
        background: transparent;
    }

    QScrollArea#pluginSwitchScroll {
        background: transparent;
        border: none;
    }

    QScrollArea#pluginSwitchScroll > QWidget > QWidget {
        background: transparent;
    }

    QFrame#workspaceCanvas {
        background: qradialgradient(cx:0.55, cy:0.3, radius:0.9,
            fx:0.55, fy:0.3, stop:0 #192434, stop:1 #0A0F16);
        border: 1px solid #202C3F;
        border-radius: 20px;
    }

    QPushButton {
        font-size: 12px;
        min-width: 96px;
        min-height: 34px;
        padding: 6px 12px;
        border-radius: 12px;
        background-color: #20486E;
        color: white;
        border: 1px solid #3A6D9C;
    }

    QPushButton:hover {
        background-color: #2A5F90;
    }

    QToolButton#layoutButton {
        font-size: 12px;
        min-width: 180px;
        min-height: 34px;
        padding: 6px 12px;
        border-radius: 12px;
        background-color: #20486E;
        color: white;
        border: 1px solid #3A6D9C;
    }

    QToolButton#layoutButton::menu-indicator {
        subcontrol-origin: padding;
        subcontrol-position: right center;
        right: 10px;
    }

    QPushButton#pluginTabButton {
        min-width: 92px;
        min-height: 32px;
        padding: 5px 14px;
        border-radius: 16px;
        background: rgba(18, 28, 42, 0.82);
        border: 1px solid #243A56;
        color: #95A8C0;
        font-size: 11px;
        font-weight: 700;
    }

    QPushButton#pluginTabButton:hover {
        background: rgba(29, 46, 68, 0.95);
        border: 1px solid #3E648E;
        color: #E8F1FF;
    }

    QPushButton#pluginTabButton:checked {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 #274765, stop:1 #356793);
        border: 1px solid #77A5D3;
        color: #F5F9FF;
    }

    QPushButton:checked {
        background-color: #2E73B5;
        border: 1px solid #74A9DA;
    }

    QPushButton:disabled {
        background-color: #324153;
        border: 1px solid #495A6F;
        color: #91A0B2;
    }

    QComboBox, QSpinBox, QLineEdit {
        min-height: 34px;
        padding: 4px 10px;
        border-radius: 10px;
        background: #0F1622;
        border: 1px solid #2A3A50;
        color: #EFF5FF;
    }

    QComboBox#modelCombo {
        background: #121B28;
        border: 1px solid #33465F;
        color: #F3F6FB;
    }

    QComboBox#modelCombo:hover {
        background: #162131;
        border: 1px solid #46617F;
    }

    QAbstractItemView#softComboPopup {
        background: #161F2B;
        border: 1px solid #31445D;
        border-radius: 10px;
        color: #E8EEF6;
        outline: none;
        selection-background-color: #28415A;
        selection-color: #F7FAFD;
    }

    QAbstractItemView#softComboPopup::item {
        min-height: 28px;
        padding: 6px 10px;
        border-radius: 8px;
        margin: 2px 4px;
    }

    QAbstractItemView#softComboPopup::item:hover {
        background: #203247;
    }

    QMenu#softMenuPopup {
        background: #161F2B;
        border: 1px solid #31445D;
        border-radius: 10px;
        color: #E8EEF6;
        padding: 4px;
    }

    QMenu#softMenuPopup::item {
        padding: 6px 10px;
        margin: 2px 4px;
        border-radius: 8px;
        background: transparent;
    }

    QMenu#softMenuPopup::item:selected {
        background: #28415A;
        color: #F7FAFD;
    }

    QMenu#softMenuPopup::separator {
        height: 1px;
        margin: 6px 8px;
        background: #31445D;
    }

    QMenu#softMenuPopup::right-arrow {
        width: 0px;
        height: 0px;
    }

    QMenu#softMenuPopup::icon {
        padding-left: 0px;
    }

    QSpinBox {
        padding-right: 42px;
    }

    QSpinBox::up-button, QSpinBox::down-button {
        subcontrol-origin: border;
        width: 24px;
        background: #162235;
        border-left: 1px solid #2A3A50;
    }

    QSpinBox::up-button {
        subcontrol-position: top right;
        border-top-right-radius: 10px;
        border-bottom: 1px solid #243754;
    }

    QSpinBox::down-button {
        subcontrol-position: bottom right;
        border-bottom-right-radius: 10px;
    }

    QSpinBox::up-button:hover, QSpinBox::down-button:hover {
        background: #21405F;
    }

    QSpinBox::up-button:pressed, QSpinBox::down-button:pressed {
        background: #2E5B86;
    }


    QSlider::groove:horizontal {
        border: 1px solid #2E405A;
        height: 6px;
        background: #162234;
        border-radius: 3px;
    }

    QSlider::handle:horizontal {
        background: #74A9DA;
        border: 1px solid #BCD9F3;
        width: 16px;
        margin: -6px 0;
        border-radius: 8px;
    }

    QCheckBox {
        spacing: 6px;
    }

    QCheckBox::indicator {
        width: 16px;
        height: 16px;
        border-radius: 4px;
        border: 1px solid #476381;
        background: #101827;
    }

    QCheckBox::indicator:checked {
        background: #4C89C9;
    }

    QListWidget#annotationList, QListWidget#volumeList {
        background: #131B26;
        border: 1px solid #2D3D52;
        border-radius: 14px;
        padding: 6px;
        outline: none;
        color: #E7EDF5;
    }

    QListWidget#annotationList::item, QListWidget#volumeList::item {
        background: #18222F;
        border: 1px solid transparent;
        border-radius: 10px;
        margin: 3px 2px;
        padding: 8px 10px;
        color: #D7E0EA;
    }

    QListWidget#annotationList::item:hover, QListWidget#volumeList::item:hover {
        background: #1D2A39;
        border: 1px solid #31455D;
    }

    QListWidget#annotationList::item:selected, QListWidget#volumeList::item:selected {
        background: #27405A;
        border: 1px solid #5C7EA1;
        color: #F6F9FC;
    }

    QLabel#sliceBadge {
        background: #142B43;
        border: 1px solid #2E5A83;
        border-radius: 10px;
        padding: 5px 10px;
        color: #CFE7FF;
        font-weight: 700;
    }

    QLabel#sliceCanvas {
        background: #090E14;
        border: 1px solid #243046;
        border-radius: 14px;
    }

    QSplitter#workspaceSplitter::handle {
        background: #162235;
        border-radius: 3px;
    }

    QFrame#slotHost {
        border: 1px solid transparent;
        border-radius: 18px;
        background: transparent;
    }

    QFrame#slotHost[dropTarget="true"] {
        border: 1px solid #74A9DA;
    }

    QFrame#viewerTile {
        background: #111722;
        border: 1px solid #31435C;
        border-radius: 16px;
    }

    TileHeader {
        background: #162235;
        border-bottom: 1px solid #243754;
        border-top-left-radius: 16px;
        border-top-right-radius: 16px;
    }

    QLabel#tileTitle {
        color: #F0F6FF;
        font-weight: 700;
        font-size: 13px;
    }

    QPushButton#leftButton, QPushButton#rightButton {
        width: 25px;
        min-width: 25px;
        max-width: 25px;
        min-height: 25px;
        max-height: 25px;
        font-size: 12px;
        border-radius: 10px;
        padding: 0px;
    }

    QPushButton#applyButton {
        width: 80px;
        min-width: 60px;
        max-width: 60px;
        min-height: 25px;
        max-height: 25px;
        font-size: 12px;
        border-radius: 10px;
        padding: 0px;
    }
"""
