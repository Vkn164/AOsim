import json

from PySide6.QtCore import Qt, Signal, Signal, Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QWidget,
    QTableWidget, QTableWidgetItem, QPushButton,
    QFileDialog, QMessageBox,
)

import scripts.utilities as ut
from data.CONFIG_DTYPES import CONFIG_DTYPES

class Config_table(QWidget):
    params_changed = Signal()

    def __init__(self, section_key, config_dict):
        super().__init__()
        self.section_key = section_key
        self.config = config_dict
        self.loading_config = False

        # parameter table
        table_layout = QVBoxLayout(self)
        self.table = QTableWidget(len(self.section_key), 2)
        self.table.setHorizontalHeaderLabels(["Parameter", "Value"])

        self.table.setWordWrap(True)

        for row, key in enumerate(self.section_key):
            value = self.config.get(key, "")

            key_item = QTableWidgetItem(" ".join(str(key).split("_")).title())
            key_item.setTextAlignment(Qt.AlignLeft | Qt.AlignTop)
            key_item.setFlags(key_item.flags() & ~Qt.ItemIsEditable)

            val_item = QTableWidgetItem(str(value))

            self.table.setItem(row, 0, key_item)
            self.table.setItem(row, 1, val_item)

        self.table.resizeRowsToContents()
        self.table.itemChanged.connect(self.param_change)

        table_layout.addWidget(self.table)

        config_save_load_layout = QHBoxLayout()
        self.config_save_b = QPushButton("Save")
        self.config_load_b = QPushButton("Load")
        config_save_load_layout.addWidget(self.config_save_b)
        config_save_load_layout.addWidget(self.config_load_b)
        table_layout.addLayout(config_save_load_layout)

        self.config_save_b.clicked.connect(self.save_file)
        self.config_load_b.clicked.connect(self.open_file)

    def param_change(self, item):
        if self.loading_config: return

        key = self.table.item(item.row(), 0).text()
        key = "_".join(str(key).lower().split())

        value_str = self.table.item(item.row(), 1).text()

        # convert to correct type
        dtype = CONFIG_DTYPES.get(key, str)  # default to string if unknown
        
        try:
            if dtype is bool:
                # handle booleans
                value = value_str.lower() in ["true", "1", "yes"]
            else:
                value = dtype(value_str)
        except Exception:
            # fallback to string if conversion fails
            value = value_str
        # update shared config
        self.config[key] = value

        ut.set_params(self.config)
        
        # emit signal so recalculation can happen
        self.params_changed.emit()

    def show_config_dtypes(config):
        text = "\n".join(f"{k}: {type(v).__name__}" for k, v in config.items())
        QMessageBox.information(None, "Config Types", text)

    def open_file(self):
        self.loading_config = True
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select config file",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        if not file_path:
            return

        with open(file_path, "r") as f:
            loaded = json.load(f)

        # update only keys in this section (if they exist in the loaded file)
        for key in self.section_key:
            if key in loaded:
                self.config[key] = loaded[key]

        # refresh table display
        for row, key in enumerate(self.section_key):
            value = self.config.get(key, "")

            key_item = QTableWidgetItem(" ".join(str(key).split("_")).title())
            key_item.setTextAlignment(Qt.AlignLeft | Qt.AlignTop)
            key_item.setFlags(key_item.flags() & ~Qt.ItemIsEditable)

            val_item = QTableWidgetItem(str(value))

            self.table.setItem(row, 0, key_item)
            self.table.setItem(row, 1, val_item)
        
        ut.set_params(self.config)
        self.params_changed.emit()
        self.loading_config = False

    def save_file(self):
        # build a dict with only the listed keys
        new_section = self.config.copy()
        for row in range(self.table.rowCount()):
            key = self.table.item(row, 0).text()
            key = "_".join(str(key).lower().split())

            value_str = self.table.item(row, 1).text()
            new_section[key] = self._convert_value(key, value_str)

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            f"Save {self.section_key} Config",
            f"{self.section_key}_config.json",
            "JSON Files (*.json)"
        )
        if not file_path:
            return

        with open(file_path, "w") as f:
            json.dump(new_section, f, indent=4)

        # update the shared config only for these keys
        for k, v in new_section.items():
            self.config[k] = v

        ut.set_params(self.config)
        self.params_changed.emit()

    @staticmethod
    def _convert_value(key, val_str):
        dtype = CONFIG_DTYPES[key]
        try:
            return dtype(val_str)
        except ValueError:
            print(val_str)
            return val_str