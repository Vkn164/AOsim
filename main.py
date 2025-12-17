import sys
import json
from pathlib import Path
import argparse

from PySide6.QtCore import Qt, QThread, QTimer, Signal, Slot, QObject, QThread, Signal, Slot, QMetaObject, Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QTabWidget,
    QTableWidget, QTableWidgetItem, QSpinBox, QSlider, QLabel, QFrame, QPushButton,
    QFileDialog, QMessageBox, QHeaderView
)

from scripts.poke_tab import Poke_tab
from scripts.turbulence_tab import Turbulence_tab

class Worker(QThread):
    finished = Signal(object)  # emit the result when done

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        result = self.func(*self.args, **self.kwargs)
        self.finished.emit(result)

import sys
class MainWindow(QMainWindow):
    # Signal to request the worker: (k, job_id)
    update_request = Signal(int, int)

    with open(Path(__file__).parent / "config.json", "r") as f:
        config = json.load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--telescope_diameter", type=float)
    parser.add_argument("--telescope_center_obscuration", type=float)
    parser.add_argument("--wfs_lambda", type=float)
    parser.add_argument("--science_lambda", type=float)
    parser.add_argument("--r0", type=float)
    parser.add_argument("--L0", type=float)
    parser.add_argument("--Vwind", type=float)
    parser.add_argument("--actuators", type=int)
    parser.add_argument("--sub_apertures", type=int)
    parser.add_argument("--frame_rate", type=int)
    parser.add_argument("--grid_size", type=int)
    parser.add_argument("--field_padding", type=int)
    parser.add_argument("--poke_amplitude", type=float)
    parser.add_argument("--random_seed", type=int)
    parser.add_argument("--use_gpu", action="store_true")  # default False
    parser.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    parser.set_defaults(use_gpu=config.get("use_gpu", False))
    parser.add_argument("--data_path", type=str)
    args, unknown = parser.parse_known_args()

    params = config.copy()
    for key, value in vars(args).items():
        if value is not None:
            params[key] = value

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AOsim")

        tabs = QTabWidget()
        tabs.setMovable(True)

        tabs.addTab(Poke_tab(self.params), "Poke Diagnostics")
        tabs.addTab(Turbulence_tab(self.params), "Turbulence")

        self.setCentralWidget(tabs)

if __name__ == "__main__":
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())