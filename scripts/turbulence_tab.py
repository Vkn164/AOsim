from PySide6.QtCore import Qt, QThread, QTimer, Signal, Slot, QThread, Signal, Slot, QMetaObject, Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QWidget,
    QTableWidget, QTableWidgetItem, QSpinBox, QSlider, QLabel, QFrame,

)

from data.CONFIG_DTYPES import CONFIG_DTYPES, enforce_config_types

class Turbulence_tab(QWidget):

    def __init__(self, config_dict):
        super().__init__()
        self.params = config_dict
