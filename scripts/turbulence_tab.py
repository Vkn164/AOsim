from PySide6.QtCore import Qt, QRect, QTimer, Signal, Slot, QThread, Signal, Slot, QMetaObject
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QWidget,
    QLabel, QFrame,
    QTabWidget, QTabBar,
    QListWidget, QListWidgetItem, QStackedWidget,
    QPushButton, QToolButton, QSizePolicy,
    QListWidget,

)

from PySide6.QtGui import QPainter

from data.CONFIG_DTYPES import CONFIG_DTYPES, enforce_config_types
from scripts.config_table import Config_table
from scripts.pgcanvas import PGCanvas
from scripts.wrap_tab import WrappedTabs
from scripts.utilities import PhaseMap_tools

class Turbulence_tab(QWidget):

    def __init__(self, config_dict):
        super().__init__()
        self.params = config_dict


        self.available_funcs = {
            "AOtools InfiniteKolmogorov": PhaseMap_tools.generate_phase_map
        }

        default_turb_params =  {k: self.params[k] for k in ["r0", "L0", "Vwind", "random_seed"] if k in self.params}
        self.turbulence_screens = [
            {"name": "AOtools InfiniteKolmogorov Default", "function": PhaseMap_tools.generate_phase_map, "data": default_turb_params}
        ]

        main_layout = QHBoxLayout(self)

        # left

        
        # ftable = QFrame()
        # ftable.setMaximumWidth(250)
        # ftable.setMinimumWidth(218)

        # ftable_layout = QVBoxLayout(ftable)
        # main_layout.addWidget(ftable)

        # middle
        fmiddle = QFrame()
        fmiddle.setFrameShape(QFrame.Box)
        fmiddle.setLineWidth(1)

        middle_layout = QVBoxLayout(fmiddle)
        self.active_canvas = PGCanvas()

        middle_layout.addWidget(self.active_canvas)

        self.turb_selector = DualListSelector(available=self.turbulence_screens, text_key="name")

        middle_layout.addWidget(self.turb_selector)


        main_layout.addWidget(fmiddle)


        # right
        right_layout = QVBoxLayout()

        fright_top = QFrame()
        fright_top.setFrameShape(QFrame.Box)
        fright_top.setLineWidth(1)

        right_top_layout = QVBoxLayout(fright_top)
        tab_widget = WrappedTabs()

        # Add tabs
        for i in range(10):
            tab_widget.add_tab(TurbulencePage(self.params, f"Very Long Tab Name {i}"))


        right_top_layout.addWidget(tab_widget)

        right_layout.addWidget(fright_top)
        main_layout.addLayout(right_layout)

class TurbulencePage(QWidget):
    def __init__(self, params, title, parent=None):
        super().__init__(parent)
        self.params = params
        self.title = title

        self.tab_frame = QFrame(self)  
        self.tab_layout = QVBoxLayout(self.tab_frame)

        self.turb_canvas = PGCanvas()
        self.tab_layout.addWidget(self.turb_canvas)

        table_config_key = ["r0", "L0", "Vwind", "random_seed"]
        self.config_table = Config_table(table_config_key, self.params)
        self.tab_layout.addWidget(self.config_table)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.tab_frame)

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

class DualListSelector(QWidget):
    def __init__(self, available=None, active=None, text_key=None, parent=None):
        super().__init__(parent)

        available = available or []
        active = active or []

        self.text_key = text_key  # can be a dict key, attribute, or function

        # Lists
        self.available_list = QListWidget()
        self.active_list = QListWidget()

        self.available_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.active_list.setSelectionMode(QListWidget.ExtendedSelection)

        # Buttons
        self.btn_add = QPushButton("→")
        self.btn_remove = QPushButton("←")
        self.btn_add.clicked.connect(self.move_to_active)
        self.btn_remove.clicked.connect(self.move_to_available)

        # Layouts
        button_layout = QVBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.btn_add)
        button_layout.addWidget(self.btn_remove)
        button_layout.addStretch()

        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Available"))
        left_layout.addWidget(self.available_list)

        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Active"))
        right_layout.addWidget(self.active_list)

        main_layout = QHBoxLayout(self)
        main_layout.addLayout(left_layout)
        main_layout.addLayout(button_layout)
        main_layout.addLayout(right_layout)

        # Populate lists
        for obj in available:
            self._add_item(self.available_list, obj)
        for obj in active:
            self._add_item(self.active_list, obj)

        # Optional: double-click moves
        self.available_list.itemDoubleClicked.connect(lambda _: self.move_to_active())
        self.active_list.itemDoubleClicked.connect(lambda _: self.move_to_available())

    # --- Internal helpers ---
    def _get_text(self, obj):
        """Determine display text for an object."""
        if self.text_key is None:
            return str(obj)
        elif callable(self.text_key):
            return str(self.text_key(obj))
        elif isinstance(obj, dict):
            return str(obj.get(self.text_key, str(obj)))
        else:
            return str(getattr(obj, self.text_key, str(obj)))

    def _add_item(self, list_widget, obj):
        item = QListWidgetItem(self._get_text(obj))
        item.setData(Qt.UserRole, obj)
        list_widget.addItem(item)

    # --- Moving items ---
    def move_to_active(self):
        for item in self.available_list.selectedItems():
            obj = item.data(Qt.UserRole)
            self._add_item(self.active_list, obj)
            self.available_list.takeItem(self.available_list.row(item))

    def move_to_available(self):
        for item in self.active_list.selectedItems():
            obj = item.data(Qt.UserRole)
            self._add_item(self.available_list, obj)
            self.active_list.takeItem(self.active_list.row(item))

    # --- Accessor methods ---
    def active_items(self):
        return [self.active_list.item(i).data(Qt.UserRole) for i in range(self.active_list.count())]

    def available_items(self):
        return [self.available_list.item(i).data(Qt.UserRole) for i in range(self.available_list.count())]




        