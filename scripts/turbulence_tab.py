from PySide6.QtCore import Qt, QTimer, Signal, Slot, QThread, Signal, QMetaObject
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QWidget,
    QLabel, QFrame,
    QListWidget, QListWidgetItem, QPushButton, QSizePolicy,
    QListWidget, QSpinBox
)

import json
from pathlib import Path

import scripts.utilities as ut
from scripts.config_table import Config_table
from scripts.pgcanvas import PGCanvas
from scripts.wrap_tab import WrappedTabs
from scripts.utilities import PhaseMap_tools
from scripts.worker import FrameWorker

import cupy as cp
import queue

class Turbulence_tab(QWidget):

    def __init__(self, config_dict):
        super().__init__()
        self.params = config_dict

        self.available_funcs = {
            "AOtools InfiniteKolmogorov": PhaseMap_tools.generate_phase_map
        } # add more in the future

        screen_directory = Path(__file__).parent.parent/ "turbulence"

        # load turbulence screens from "/turbulence" folder
        self.turbulence_screens = []
        for file_path in screen_directory.iterdir():
            if file_path.is_file():   # skip subdirectories
                with file_path.open("r") as f:
                    content = json.load(f)       
                    content_name = file_path.stem
                    content_data = {k: v for k, v in content.items() if k != "function"}
                    self.turbulence_screens.append({"name": content_name, "data": content_data, "function": self.available_funcs[content["function"]]})   

        # initalize tab/page for each loaded screen
        self.turbPages = []
        for i, screen in enumerate(self.turbulence_screens):
            turbPage = TurbulencePage(screen["function"], screen["data"], screen["name"])
            self.turbPages.append(turbPage)

        main_layout = QHBoxLayout(self)

        # left -- final psf viewer
        left_layout = QVBoxLayout()
        fleft = QFrame()
        fleft.setFrameShape(QFrame.Box)
        fleft.setLineWidth(1)
        left_lay1 = QVBoxLayout(fleft)
        self.total_canvas = PGCanvas()

        left_lay1.addWidget(self.total_canvas)
        left_lay1.addWidget(QLabel("Placeholder"))

        left_layout.addWidget(fleft)

        func_sel_layout = QHBoxLayout()

        self.func_selector = QListWidget()
        self.func_selector.addItems(list(self.available_funcs.keys()))
        self.func_selector.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        func_sel_layout.addWidget(self.func_selector)

        func_options_layout = QVBoxLayout()
        for i in range(5):
            func_options_layout.addWidget(QLabel(f"Placeholder {i}"))

        func_sel_layout.addLayout(func_options_layout)


        left_layout.addLayout(func_sel_layout)

        main_layout.addLayout(left_layout)

        # middle -- layered turbulence viewer
        fmiddle = QFrame()
        fmiddle.setFrameShape(QFrame.Box)
        fmiddle.setLineWidth(1)

        ## plot canvas
        middle_layout = QVBoxLayout(fmiddle)
        self.active_canvas = PGCanvas()

        middle_layout.addWidget(self.active_canvas)

        phase_b_layout = QHBoxLayout()
        middle_layout.addLayout(phase_b_layout)

        ## run/stop buttons
        self.next_button = QPushButton("Next Step")
        self.run_x_button = QPushButton("Run X Frames")
        self.run_inf_button = QPushButton("Run Indefinitely")
        self.spin = QSpinBox()
        self.spin.setRange(1, 1000)
        self.spin.setValue(10)
        self.stop_button = QPushButton("Stop")
        self.reset_button = QPushButton("Reset")

        phase_b_layout.addWidget(self.next_button)
        phase_b_layout.addWidget(self.run_x_button)
        phase_b_layout.addWidget(self.spin)
        phase_b_layout.addWidget(self.run_inf_button)
        phase_b_layout.addWidget(self.stop_button)
        phase_b_layout.addWidget(self.reset_button)

        self.next_button.clicked.connect(self.step_active)
        self.run_x_button.clicked.connect(self.run_x_active)
        self.run_inf_button.clicked.connect(self.run_active)
        self.stop_button.clicked.connect(self.stop_active)
        self.reset_button.clicked.connect(self.reset_active)
 
        ## list selectors for active/inactive screens
        self.turb_selector = DualListSelector(available=self.turbPages[1:], active=self.turbPages[:1], text_key="title")

        self.turb_selector.activeChanged.connect(self.plot_active)
        self.func_selector.itemDoubleClicked.connect(lambda it: self.turb_selector._add_item(self.turb_selector.available_list, self.load_turb(it.text())))

        middle_layout.addWidget(self.turb_selector)


        main_layout.addWidget(fmiddle)

 
        # right -- turbulence editor
        right_layout = QVBoxLayout()

        fright_top = QFrame()
        fright_top.setFrameShape(QFrame.Box)
        fright_top.setLineWidth(1)

        right_top_layout = QVBoxLayout(fright_top)
        self.tab_widget = WrappedTabs()
        self.tab_widget.tabKilled.connect(self.turb_selector.remove_item)

        # Add tabs
        for i, page in enumerate(self.turbPages):
            self.tab_widget.add_tab(page)

        right_top_layout.addWidget(self.tab_widget)

        right_layout.addWidget(fright_top)
        main_layout.addLayout(right_layout)

        QTimer.singleShot(0, self.start_turbulence_worker)

    def load_turb(self, func):
        content = None
        content_name = None
        content_data = None
        file_path = (Path(__file__).parent.parent/ "turbulence" / "defaults" / (str(func) + ".json"))
        if file_path.is_file():   # skip subdirectories
            with file_path.open("r") as f:
                content = json.load(f)       
                content_name = file_path.stem
                content_data = {k: v for k, v in content.items() if k != "function"}

        page = TurbulencePage(self.available_funcs[content["function"]], content_data, content_name)

        page = self.check_dup(page)
        self.turbulence_screens.append({"name": page.title, "data": page.params, "function": page.function})
        
        self.turbPages.append(page)

        self.tab_widget.add_tab(page)
        
        self.create_page_thread(page)

        return page
    
    def check_dup(self, new_page):
        existing_titles = [tp.title for tp in self.turbPages]

        base_title = new_page.title
        counter = 1
        while new_page.title in existing_titles:
            new_page.title = f"{base_title} ({counter})"
            counter += 1

        return new_page

    # create thread for each loaded screen
    ## NOTE need to change once ability to add/delete screens implemented
    def start_turbulence_worker(self):
        self._frame_queue = queue.Queue()
        self._threads = []

        for turbPage in self.turbPages:
            self.create_page_thread(turbPage)

        # timer runs in GUI thread
        self._frame_timer = QTimer(self)
        self._frame_timer.timeout.connect(self._process_frames)
        self._frame_timer.start(1000/30)

    def create_page_thread(self, page):
        worker = FrameWorker(page.function, page.params)
        thread = QThread()

        worker.moveToThread(thread)

        # add created frame to queue once done
        worker.frame_ready.connect(
            lambda f, p=page: self._frame_queue.put((p, f))
        )
        
        thread.start()
        thread.started.connect(worker.step) # initialize threads/pages on startup

        page._worker = worker
        page._thread = thread
        self._threads.append((thread, worker))

   
    # update pages and layered screen viewer when threads finish
    def _process_frames(self):
        updated = False

        while not self._frame_queue.empty():
            page, frame = self._frame_queue.get()
            page.current_screen = frame

            updated = True

            if page.isVisible():
                page.turb_canvas.set_image(frame)

        if updated:
            self.plot_active()

    # get currently active screens
    def get_active_screens(self):
        active = [
                    item.current_screen for item in self.turb_selector.active_items()
                    if hasattr(item, "current_screen")
                 ]
        active = cp.asarray(active)
        return cp.sum(active, axis=0)
    
    def plot_active(self):
        total_active = self.get_active_screens()
        self.active_canvas.set_image(total_active)

        # calculate final PSF for active screens
        self.total_science, self.total_science_strehl = ut.Analysis.generate_science_image(phase_map=total_active)
        self.normalized_image = self.total_science/self.total_science.sum()
        self.total_science_plot = cp.log10(self.normalized_image + 1e-12)
        self.total_canvas.set_image(self.total_science_plot)

    # for run/stop buttons
    def run_active(self):
        for page in self.turb_selector.active_items():
            page.run()

    def stop_active(self):
        for page in self.turb_selector.active_items():
            page.stop()

    def step_active(self):
        for page in self.turb_selector.active_items():
            page.step_once()

    def run_x_active(self, n):
        for page in self.turb_selector.active_items():
            page.run_x(n)

    def reset_active(self):
        for page in self.turb_selector.active_items():
            page.reset()



class TurbulencePage(QWidget):
    new_frame = Signal(object) 

    def __init__(self, function, params, title, parent=None):
        super().__init__(parent)
        self.function = function
        self.params = params
        self.title = title

        self.current_screen = None
        self.displayed = False

        self._worker = None
        self._thread = None


        # layout
        tab_frame = QFrame(self)  
        tab_layout = QVBoxLayout(tab_frame)

        # screen view
        self.turb_canvas = PGCanvas()

        tab_layout.addWidget(self.turb_canvas)

        # --- Controls ---
        phase_b_layout = QHBoxLayout()
        tab_layout.addLayout(phase_b_layout)

        self.next_button = QPushButton("Next Step")
        self.run_x_button = QPushButton("Run X Frames")
        self.run_inf_button = QPushButton("Run Indefinitely")
        self.spin = QSpinBox()
        self.spin.setRange(1, 1000)
        self.spin.setValue(10)
        self.stop_button = QPushButton("Stop")
        self.reset_button = QPushButton("Reset")

        phase_b_layout.addWidget(self.next_button)
        phase_b_layout.addWidget(self.run_x_button)
        phase_b_layout.addWidget(self.spin)
        phase_b_layout.addWidget(self.run_inf_button)
        phase_b_layout.addWidget(self.stop_button)
        phase_b_layout.addWidget(self.reset_button)

        self.next_button.clicked.connect(self.page_next_button_clicked)
        self.run_x_button.clicked.connect(self.page_run_x_button_clicked)
        self.run_inf_button.clicked.connect(self.page_run_button_clicked)
        self.stop_button.clicked.connect(self.page_stop_button_clicked)
        self.reset_button.clicked.connect(self.page_reset_button_clicked)


        # --- Config table ---
        table_config_key = list(self.params.keys())
        self.config_table = Config_table(table_config_key, self.params)
        tab_layout.addWidget(self.config_table)

        self.config_table.params_changed.connect(lambda pc: setattr(self, "params", pc))
        self.config_table.params_changed.connect(self.page_reset_button_clicked)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(tab_frame)

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

    # refresh viewer when tab/page is shown
    def showEvent(self, event):
        super().showEvent(event)

        if self.displayed and self.current_screen is not None:
            self.turb_canvas.set_image(self.current_screen)

    def page_next_button_clicked(self):
        if hasattr(self, "_worker"):
            QMetaObject.invokeMethod(self._worker, "step", Qt.QueuedConnection)

    def page_run_x_button_clicked(self):
        if hasattr(self, "_worker"):
            for i in range(self.spin.value()):
                QMetaObject.invokeMethod(self._worker, "step", Qt.QueuedConnection)

    def page_run_button_clicked(self):
        if hasattr(self, "_worker"):
            QMetaObject.invokeMethod(self._worker, "run", Qt.QueuedConnection)

    def page_stop_button_clicked(self):
        if hasattr(self, "_worker"):
            QMetaObject.invokeMethod(self._worker, "stop", Qt.QueuedConnection)

    def page_reset_button_clicked(self):
        # stop worker then recreate generator inside worker thread:
        if hasattr(self, "_worker"):
            QMetaObject.invokeMethod(self._worker, "stop", Qt.QueuedConnection)
            QMetaObject.invokeMethod(self._worker, "reset", Qt.QueuedConnection)

    # allow for main widget to activate threads
    @Slot()
    def stop(self):
        self.page_stop_button_clicked()

    @Slot()
    def step_once(self):
        self.page_next_button_clicked() 
        
    @Slot()
    def run(self):
        self.page_run_button_clicked()

    @Slot(int)
    def run_x(self, n):
        if hasattr(self, "_worker"):
            for i in range(self.spin.value()):
                QMetaObject.invokeMethod(self._worker, "step", Qt.QueuedConnection)

    @Slot()
    def reset(self):
        self.page_reset_button_clicked() 

    # ---- kill page ----
    def kill(self):
        # stop any loops/workers
        if hasattr(self, "_worker"):
            self._worker.stop()

        if hasattr(self, "_thread"):
            self._thread.quit()
            self._thread.wait(1000) 

    def cleanup(self):
        try:
            self.new_frame.disconnect()
        except TypeError:
            pass




class DualListSelector(QWidget):
    activeChanged = Signal(list)
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

    def remove_item(self, obj):
        removed_from_active = False

        for lst in (self.available_list, self.active_list):
            for i in reversed(range(lst.count())):
                item = lst.item(i)
                if item.data(Qt.UserRole) is obj:
                    lst.takeItem(i)
                    if lst is self.active_list:
                        removed_from_active = True

        if removed_from_active:
            self.activeChanged.emit(self.active_items())

    # --- Moving items ---
    def move_to_active(self):
        for item in self.available_list.selectedItems():
            obj = item.data(Qt.UserRole)
            self._add_item(self.active_list, obj)
            self.available_list.takeItem(self.available_list.row(item))

        self.activeChanged.emit(self.active_items)

    def move_to_available(self):
        for item in self.active_list.selectedItems():
            obj = item.data(Qt.UserRole)
            self._add_item(self.available_list, obj)
            self.active_list.takeItem(self.active_list.row(item))

        self.activeChanged.emit(self.active_items)

    # --- Accessor methods ---
    def active_items(self):
        return [self.active_list.item(i).data(Qt.UserRole) for i in range(self.active_list.count())]

    def available_items(self):
        return [self.available_list.item(i).data(Qt.UserRole) for i in range(self.available_list.count())]




