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
    params_changed = Signal(dict)

    def __init__(self, section_key, config_dict):
        super().__init__()
        self.section_key = list(section_key)          # original keys in order
        self.config = config_dict
        self.loading_config = False

        # mapping: readable header -> original key
        self.header_map = {}
        # row index -> original key (fast lookup)
        self.row_keys = list(self.section_key)

        # parameter table
        table_layout = QVBoxLayout(self)
        self.table = QTableWidget(len(self.row_keys), 2)
        self.table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.table.setWordWrap(True)

        for row, key in enumerate(self.row_keys):
            value = self.config.get(key, "")

            readable = " ".join(str(key).split("_")).title()
            # save mapping
            self.header_map[readable] = key

            key_item = QTableWidgetItem(readable)
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
        # only respond to edits in the Value column
        if self.loading_config:
            return
        if item.column() != 1:
            return

        row = item.row()
        # get original key for this row
        if row < 0 or row >= len(self.row_keys):
            return
        key = self.row_keys[row]

        value_str = self.table.item(row, 1).text()

        # convert to correct type (default to string)
        dtype = CONFIG_DTYPES.get(key, str)

        try:
            if dtype is bool:
                value = value_str.lower() in ["true", "1", "yes"]
            else:
                value = dtype(value_str)
        except Exception:
            value = value_str

        # update shared config and emit
        self.config[key] = value
        self.params_changed.emit(self.config)

    def show_config_dtypes(self):
        text = "\n".join(f"{k}: {type(v).__name__}" for k, v in self.config.items())
        QMessageBox.information(self, "Config Types", text)

    def open_file(self):
        self.loading_config = True
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select config file",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        if not file_path:
            self.loading_config = False
            return

        with open(file_path, "r") as f:
            loaded = json.load(f)

        # update only keys in this section (if they exist in the loaded file)
        for key in self.row_keys:
            if key in loaded:
                self.config[key] = loaded[key]

        # refresh table display (use existing row_keys order)
        for row, key in enumerate(self.row_keys):
            value = self.config.get(key, "")

            readable = " ".join(str(key).split("_")).title()
            key_item = QTableWidgetItem(readable)
            key_item.setTextAlignment(Qt.AlignLeft | Qt.AlignTop)
            key_item.setFlags(key_item.flags() & ~Qt.ItemIsEditable)

            val_item = QTableWidgetItem(str(value))

            # block signals while updating items to avoid param_change firing
            self.table.blockSignals(True)
            self.table.setItem(row, 0, key_item)
            self.table.setItem(row, 1, val_item)
            self.table.blockSignals(False)

        self.params_changed.emit(self.config)
        self.loading_config = False

    def save_file(self):
        # build a dict with only the listed keys (in this section)
        new_section = {}
        for row in range(self.table.rowCount()):
            key_readable = self.table.item(row, 0).text()
            orig_key = self.header_map.get(key_readable, "_".join(key_readable.lower().split()))
            value_str = self.table.item(row, 1).text()
            new_section[orig_key] = self._convert_value(orig_key, value_str)

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

        self.params_changed.emit(self.config)

    @staticmethod
    def _convert_value(key, val_str):
        dtype = CONFIG_DTYPES.get(key, str)
        try:
            if dtype is bool:
                return val_str.lower() in ["true", "1", "yes"]
            return dtype(val_str)
        except Exception:
            return val_str
