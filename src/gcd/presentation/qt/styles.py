DARK_STYLESHEET = """
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

    QComboBox#modelCombo, QComboBox#softCombo {
        background: #121B28;
        border: 1px solid #33465F;
        color: #F3F6FB;
    }

    QComboBox#modelCombo:hover, QComboBox#softCombo:hover {
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

    QFrame#tileHeader {
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

    QToolButton#themeToggleButton {
        min-width: 38px;
        max-width: 38px;
        min-height: 38px;
        max-height: 38px;
        border-radius: 19px;
        background: #142032;
        border: 1px solid #2A3850;
        color: #FFE08A;
        font-size: 18px;
        padding: 0px;
    }

    QToolButton#themeToggleButton:hover {
        background: #1D314A;
        border: 1px solid #4C7EB8;
    }
"""


CLASSIC_STYLESHEET = """
    QWidget {
        color: #1F1F1F;
        font-family: "Segoe UI";
        font-size: 13px;
    }

    QMainWindow, QWidget#centralwidget {
        background: #F3F3F3;
    }

    QLabel {
        color: #1F1F1F;
        background: transparent;
    }

    QLabel:disabled {
        color: #8A8A8A;
    }

    QFrame#globalToolbar, QFrame#workspaceToolbar {
        background: #FFFFFF;
        border: 1px solid #E5E5E5;
        border-radius: 8px;
    }

    QLabel#appTitle {
        font-size: 18px;
        font-weight: 600;
        color: #1F1F1F;
    }

    QLabel#appSubtitle,
    QLabel#panelDescription,
    QLabel#workspaceHint,
    QLabel#inspectorKicker {
        color: #5F5F5F;
        font-size: 12px;
    }

    QLabel#panelTitle {
        font-size: 16px;
        font-weight: 600;
        color: #1F1F1F;
    }

    QLabel#statusPill,
    QLabel#sliceBadge {
        padding: 5px 10px;
        border-radius: 6px;
        background: #F9F9F9;
        border: 1px solid #E5E5E5;
        color: #1F1F1F;
        font-weight: 600;
    }

    QFrame#pluginStack,
    QStackedWidget#pluginStack,
    QFrame#inspectorHeader,
    QListWidget#pluginRail,
    QFrame#workspaceCanvas,
    QFrame#viewerTile,
    QFrame#tileHeader,
    QFrame#slotHost {
        background: #FFFFFF;
        border: 1px solid #E5E5E5;
        border-radius: 8px;
    }

    QFrame#workspaceCanvas {
        background: #FAFAFA;
    }

    QFrame#viewerTile {
        background: #FFFFFF;
        border: 1px solid #E5E5E5;
    }

    QFrame#tileHeader {
        background: #F9F9F9;
        border-bottom: 1px solid #E5E5E5;
    }

    QLabel#tileTitle {
        color: #1F1F1F;
        font-weight: 600;
        font-size: 13px;
    }

    QWidget#pluginSwitchStrip,
    QWidget#pluginSwitchContent,
    QFrame#pluginSwitchRow {
        background: transparent;
        border: none;
    }

    QScrollArea#pluginSwitchScroll {
        background: transparent;
        border: none;
    }

    QScrollArea#pluginSwitchScroll > QWidget > QWidget {
        background: transparent;
    }

    QGroupBox {
        background: #FFFFFF;
        border: 1px solid #E5E5E5;
        border-radius: 8px;
        margin-top: 18px;
        padding: 10px;
        color: #1F1F1F;
        font-weight: 600;
    }

    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 10px;
        padding: 0px 4px;
        color: #1F1F1F;
        background: #FFFFFF;
    }

    QGroupBox:disabled,
    QGroupBox::title:disabled {
        color: #8A8A8A;
        border-color: #EAEAEA;
    }

    QPushButton {
        font-size: 12px;
        min-width: 96px;
        min-height: 32px;
        padding: 5px 12px;
        border-radius: 4px;
        background-color: #0067C0;
        color: #FFFFFF;
        border: 1px solid #005A9E;
        font-weight: 500;
    }

    QPushButton:hover {
        background-color: #005A9E;
        border: 1px solid #004E8C;
    }

    QPushButton:disabled {
        background-color: #F5F5F5;
        border: 1px solid #E0E0E0;
        color: #9A9A9A;
    }

    QPushButton:checked,
    QPushButton#pluginTabButton:checked {
        background-color: #E5F1FB;
        border: 1px solid #0067C0;
        color: #003E73;
    }

    QPushButton#pluginTabButton {
        min-width: 92px;
        min-height: 32px;
        padding: 5px 12px;
        border-radius: 16px;
        background-color: #F9F9F9;
        color: #1F1F1F;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid #E5E5E5;
    }

    QPushButton#pluginTabButton:hover {
        background-color: #F0F6FC;
        border: 1px solid #C7E0F4;
    }

    QPushButton#pluginTabButton:disabled {
        background-color: #F5F5F5;
        border: 1px solid #E0E0E0;
        color: #9A9A9A;
    }

    QToolButton#layoutButton {
        font-size: 12px;
        min-width: 132px;
        min-height: 32px;
        padding: 5px 12px;
        border-radius: 4px;
        background-color: #FFFFFF;
        color: #1F1F1F;
        border: 1px solid #DADADA;
    }

    QToolButton#layoutButton::menu-indicator {
        subcontrol-origin: padding;
        subcontrol-position: right center;
        right: 8px;
    }

    QComboBox, QSpinBox, QLineEdit {
        min-height: 32px;
        padding: 4px 9px;
        border-radius: 4px;
        background: #FFFFFF;
        border: 1px solid #DADADA;
        color: #1F1F1F;
    }

    QComboBox#modelCombo, QComboBox#softCombo {
        background: #FFFFFF;
        border: 1px solid #DADADA;
        color: #1F1F1F;
    }

    QComboBox:hover, QSpinBox:hover, QLineEdit:hover,
    QComboBox#modelCombo:hover, QComboBox#softCombo:hover {
        border: 1px solid #B8B8B8;
        background: #FFFFFF;
    }

    QComboBox:focus, QSpinBox:focus, QLineEdit:focus {
        border: 1px solid #0067C0;
    }

    QComboBox:disabled,
    QSpinBox:disabled,
    QLineEdit:disabled {
        background: #F5F5F5;
        border: 1px solid #E0E0E0;
        color: #9A9A9A;
    }

    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: right center;
        width: 24px;
        border-left: 1px solid #E5E5E5;
        border-top-right-radius: 4px;
        border-bottom-right-radius: 4px;
        background: #F9F9F9;
    }

    QComboBox::drop-down:hover {
        background: #F0F6FC;
    }

    QComboBox::down-arrow {
        width: 0px;
        height: 0px;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 5px solid #4A4A4A;
        margin-right: 8px;
    }

    QComboBox QAbstractItemView,
    QAbstractItemView#softComboPopup,
    QMenu#softMenuPopup {
        background: #FFFFFF;
        border: 1px solid #DADADA;
        border-radius: 8px;
        color: #1F1F1F;
        outline: none;
        selection-background-color: #E5F1FB;
        selection-color: #1F1F1F;
    }

    QComboBox QAbstractItemView::item,
    QAbstractItemView#softComboPopup::item {
        min-height: 28px;
        padding: 5px 8px;
        color: #1F1F1F;
    }

    QMenu#softMenuPopup::item:selected {
        background: #E5F1FB;
        color: #1F1F1F;
    }

    QSpinBox::up-button,
    QSpinBox::down-button {
        background: #F6F6F6;
        border-left: 1px solid #DADADA;
    }

    QSpinBox::up-button:hover,
    QSpinBox::down-button:hover {
        background: #F0F6FC;
    }

    QLineEdit {
        min-width: 100px;
    }

    QTextEdit {
        background-color: #FFFFFF;
        color: #1F1F1F;
        font-family: Consolas;
        font-size: 12px;
        border: 1px solid #DADADA;
        border-radius: 4px;
    }

    QSlider::groove:horizontal {
        border: none;
        height: 4px;
        background: #DADADA;
        border-radius: 2px;
    }

    QSlider::handle:horizontal {
        background: #0067C0;
        border: 2px solid #FFFFFF;
        width: 18px;
        margin: -8px 0;
        border-radius: 9px;
    }

    QCheckBox::indicator {
        width: 16px;
        height: 16px;
        border-radius: 3px;
        border: 1px solid #8A8A8A;
        background: #FFFFFF;
    }

    QCheckBox::indicator:checked {
        background: #0067C0;
        border: 1px solid #0067C0;
    }

    QListWidget#annotationList, QListWidget#volumeList {
        background: #FFFFFF;
        border: 1px solid #DADADA;
        border-radius: 8px;
        padding: 4px;
        outline: none;
        color: #1F1F1F;
    }

    QListWidget#annotationList::item, QListWidget#volumeList::item {
        background: #FFFFFF;
        border: none;
        border-radius: 4px;
        margin: 2px;
        padding: 6px 8px;
        color: #1F1F1F;
    }

    QListWidget#annotationList::item:hover, QListWidget#volumeList::item:hover {
        background: #F5F5F5;
    }

    QListWidget#annotationList::item:selected, QListWidget#volumeList::item:selected {
        background: #E5F1FB;
        color: #003E73;
    }

    QWidget#sliceCanvas {
        background: #111111;
        border: 1px solid #DADADA;
        border-radius: 8px;
    }

    QSplitter#workspaceSplitter::handle {
        background: #E5E5E5;
        border-radius: 2px;
    }

    QFrame#slotHost[dropTarget="true"] {
        border: 1px solid #0067C0;
    }

    QPushButton#leftButton, QPushButton#rightButton {
        width: 25px;
        min-width: 25px;
        max-width: 25px;
        min-height: 25px;
        max-height: 25px;
        font-size: 12px;
        border-radius: 4px;
        padding: 0px;
    }

    QPushButton#applyButton {
        width: 80px;
        min-width: 60px;
        max-width: 60px;
        min-height: 25px;
        max-height: 25px;
        font-size: 12px;
        border-radius: 4px;
        padding: 0px;
    }

    QToolButton#themeToggleButton {
        min-width: 38px;
        max-width: 38px;
        min-height: 38px;
        max-height: 38px;
        border-radius: 19px;
        background: #FFFFFF;
        border: 1px solid #DADADA;
        color: #1F1F1F;
        font-size: 18px;
        padding: 0px;
    }

    QToolButton#themeToggleButton:hover {
        background: #F5F5F5;
        border: 1px solid #B8B8B8;
    }
"""


STYLESHEET = DARK_STYLESHEET
