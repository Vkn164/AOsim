from PySide6.QtCore import Qt, QThread, QTimer, Signal, Slot, QThread, Signal, Slot, QMetaObject, Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QWidget,
    QTableWidget, QTableWidgetItem, QSpinBox, QSlider, QLabel, QFrame,

)

import cupy as cp
import numpy as np

import scripts.utilities as ut
from scripts.pgcanvas import PGCanvas
from scripts.worker import CalculateWorker
from scripts.config_table import Config_table
from data.CONFIG_DTYPES import CONFIG_DTYPES, enforce_config_types

class Poke_tab(QWidget):
    update_request = Signal(int, int)

    def __init__(self, config_dict):
        super().__init__()
        self.params = config_dict

        # calculation jobs bookkeeping
        self.job_id = 0
        self.pending_index = 0

        # layout for entire tab
        main_layout = QVBoxLayout(self)

        ## top half
        top_layout = QHBoxLayout()


        ## top left -- parameter configuration
        ftable = QFrame()
        ftable.setMaximumWidth(250)
        ftable.setMinimumWidth(218)

        ftable_layout = QVBoxLayout(ftable)
        table_config_key = list(self.params.keys())[:9]
        self.config_table = Config_table(table_config_key, self.params)
        
        ftable_layout.addWidget(self.config_table)

        ## Busy indicator
        self.busy_label = QLabel("Waiting")
        self.busy_label.setAlignment(Qt.AlignCenter)
        self.busy_label.setStyleSheet("font-weight: bold;")

        ftable_layout.addWidget(self.busy_label)


        top_layout.addWidget(ftable)


        ## middle: overview canvas
        fmiddle_v = QFrame()
        fmiddle_v.setFrameShape(QFrame.Box)
        fmiddle_v.setLineWidth(1)
        middle_v = QVBoxLayout(fmiddle_v)
        middle_v.addWidget(QLabel("Sub Aperture Centroids"))

        self.canvas_overview = PGCanvas()
        middle_v.addWidget(self.canvas_overview)
        top_layout.addWidget(fmiddle_v)

        # right: selector + subap canvas
        fright_v = QFrame()
        fright_v.setFrameShape(QFrame.Box)
        fright_v.setLineWidth(1)
        right_v = QVBoxLayout(fright_v)
        right_v.addWidget(QLabel("Sub Aperture Images"))

        # aperture selector
        selector_h = QHBoxLayout()
        act_selection_label = QLabel("Poke Actuator Idx")
        self.act_select_spinbox = QSpinBox()
        self.act_select_slider = QSlider(Qt.Horizontal)
        selector_h.addWidget(act_selection_label)
        selector_h.addWidget(self.act_select_spinbox)
        selector_h.addWidget(self.act_select_slider)
        right_v.addLayout(selector_h)

        self.canvas_subap = PGCanvas()
        right_v.addWidget(self.canvas_subap)
        top_layout.addWidget(fright_v)

        main_layout.addLayout(top_layout)



        # bottom layout
        bottom_layout = QHBoxLayout()

        # bottom left -- calculated values table
        fcalc_vals = QFrame()
        calc_values = QVBoxLayout(fcalc_vals)
        self.calc_values_table = QTableWidget(5, 2)
        self.calc_values_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.calc_values_table.setEditTriggers(QTableWidget.NoEditTriggers)

        self.calc_values_table.setItem(0, 0, QTableWidgetItem("Strehl"))
        self.calc_values_table.setItem(1, 0, QTableWidgetItem("Poke FWHM (mrad)"))
        self.calc_values_table.setItem(2, 0, QTableWidgetItem("Unperturbed FWHM (mrad)"))
        #self.calc_values_table.setItem(4, 0, QTableWidgetItem("Unperturbed Peak Intensity"))
        self.calc_values_table.setItem(3, 0, QTableWidgetItem("Diffraction Limit (mrad)"))

        self.calc_values_table.setWordWrap(True)

        for row in range(self.calc_values_table.rowCount()):
            item = self.calc_values_table.item(row, 0)
            if item is not None:
                item.setTextAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.calc_values_table.resizeRowsToContents()
        self.calc_values_table.setMinimumWidth(200)


        calc_values.addWidget(self.calc_values_table)

        fcalc_vals.setMaximumWidth(250)
        bottom_layout.addWidget(fcalc_vals)       

        
        # science image w poke canvas
        fmiddle_vb = QFrame()
        fmiddle_vb.setFrameShape(QFrame.Box)
        fmiddle_vb.setLineWidth(1)
        middle_vb = QVBoxLayout(fmiddle_vb)
        middle_vb.addWidget(QLabel("Poke Science Image"))

        self.canvas_science = PGCanvas()
        middle_vb.addWidget(self.canvas_science)
        bottom_layout.addWidget(fmiddle_vb)

        # science image canvas
        fright_vb = QFrame()
        fright_vb.setFrameShape(QFrame.Box)
        fright_vb.setLineWidth(1)
        right_vb = QVBoxLayout(fright_vb)
        right_vb.addWidget(QLabel("Unperturbed Science Image"))

        self.canvas_science_pupil = PGCanvas()
        right_vb.addWidget(self.canvas_science_pupil)
        bottom_layout.addWidget(fright_vb)

        main_layout.addLayout(bottom_layout)


        # calculation worker/thread init
        self.calc_thread = None
        self.calc_worker = None

        # Prepare AO data (start background calculation)
        self.start_calculation()

        # Timers for actuator selector
        # prevents frequent updates while actuator slider is rapidly changing
        self.debounce_timer = QTimer(self)
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.setInterval(40)  # ms
        self.debounce_timer.timeout.connect(self._emit_worker_request)

        self.act_select_slider.valueChanged.connect(self._on_spin_changed)
        self.act_select_spinbox.valueChanged.connect(self._on_spin_changed)

        # keep act_select_slider/act_select_spinbox in sync
        self.act_select_spinbox.valueChanged.connect(self.act_select_slider.setValue)
        self.act_select_slider.valueChanged.connect(self.act_select_spinbox.setValue)

        # if params change, recalculate, also change utilities functions default parameters too
        self.config_table.params_changed.connect(self.recalculate)
        self.config_table.params_changed.connect(ut.set_params)

        # initial request on start
        self._schedule_update(0)

    def start_calculation(self):
        # stop existing calc thread if running
        if hasattr(self, "calc_thread") and self.calc_thread is not None and self.calc_thread.isRunning():
            try:
                self.calc_thread.quit()
                self.calc_thread.wait(10)
            except Exception:
                pass

        # create worker + thread
        self.calc_thread = QThread(self)
        self.calc_worker = CalculateWorker(self.params)
        self.calc_worker.moveToThread(self.calc_thread)

        # connect lifecycle
        self.calc_thread.started.connect(self.calc_worker.initialize)  # run initial compute when thread starts
        self.calc_thread.started.connect(lambda: self.busy_label.setText("Calculating"))

        self.calc_worker.initialized_result.connect(self.on_calculation_finished)
        self.calc_worker.initialized_result.connect(lambda _: self.busy_label.setText("Waiting"))

        # connect per-actuator responses
        self.calc_worker.subap_finished.connect(self.on_worker_finished)
        # cleanup
        self.calc_worker.finished.connect(self.calc_thread.quit)
        self.calc_worker.finished.connect(self.calc_worker.deleteLater)
        self.calc_thread.finished.connect(self.calc_thread.deleteLater)


        # start thread (this will call initialize())
        self.calc_thread.start()

        try:
            self.update_request.disconnect()
        except Exception:
            pass
        self.update_request.connect(self.calc_worker.process_subap)

    @Slot(object)
    def on_calculation_finished(self, result):
        """Main-thread handler: update GUI with the initial/full calculation result."""
        # store arrays & metadata on self (copied references; CPU/GPU arrays are fine)
        self.busy_label.setText("Calculation Finished")

        self.pupil = result["pupil"]
        self.act_centers = result["act_centers"]
        self.influence_maps = result["influence_maps"]
        self.active_sub_aps = result["active_sub_aps"]
        self.sub_aps = result["sub_aps"]
        self.sub_aps_idx = result["sub_aps_idx"]
        self.sub_ap_width = result["sub_ap_width"]
        self.sub_slice = result["sub_slice"]
        self.sub_pupils = result["sub_pupils"]
        self.ref_centroids = result["ref_centroids"]
        self.ref_images = result["ref_images"]
        self.ref_science_image = result["ref_science_image"]
        self.ref_strehl = result["ref_strehl"]
        self.normalized_image = result["normalized_image"]

        # update the unperturbed science canvas
        self.canvas_science_pupil.set_image(result["science_image_plot"])

        # update table values
        ref_fwhm = result["ref_fwhm"]
        diff_lim = result["diff_lim"]
        self.calc_values_table.setItem(2, 1, QTableWidgetItem(f"{ref_fwhm*1e3:.3e}"))
        self.calc_values_table.setItem(3, 1, QTableWidgetItem(f"{diff_lim*1e3:.3e}"))

        # set actuator ranges
        n_act = result["n_act"]
        self.act_select_spinbox.setRange(0, n_act - 1)
        self.act_select_slider.setRange(0, n_act - 1)
        self.act_select_slider.setSingleStep(1)


    @Slot(object, int)
    def on_worker_finished(self, result, job_id):
        """result is (centroids, subap_images) and job_id matches what was emitted."""
        enforce_config_types(self.params)
        # ignore stale results
        if job_id != self.job_id:
            return

        self.busy_label.setText("Plotting")

        # Plot update
        k = int(self.pending_index)  # current actuator index
        centroids, subap_images = result

        influence_phase_scale = 4 * cp.pi / float(self.params.get("science_lambda"))
        influence_map_phase = self.influence_maps[k] * self.params.get("poke_amplitude") * influence_phase_scale

        # update science image canvas
        self.science_image, self.strehl = ut.Analysis.generate_science_image(self.pupil, influence_map_phase)
        self.normalized_image = self.science_image/self.science_image.sum()
        self.science_image_plot = cp.log10(self.normalized_image + 1e-12)
        self.canvas_science.set_image(self.science_image_plot)

        # update overview canvas
        self.canvas_overview.set_image_masked(self.influence_maps[k].get(), mask=self.pupil.get())
        self.canvas_overview.set_points_and_quivers(self.ref_centroids + self.sub_aps - self.sub_ap_width/4,
                                                   centroids + self.sub_aps - self.sub_ap_width/4)

        plate_rad = (self.params.get("science_lambda") * self.params.get("grid_size")
                     / (self.params.get("telescope_diameter") * self.science_image.shape[0]))
        img_fwhm = plate_rad * ut.Analysis.fwhm_radial(self.science_image)
        self.calc_values_table.setItem(0, 1, QTableWidgetItem(f"{self.strehl:.3f}"))
        self.calc_values_table.setItem(1, 1, QTableWidgetItem(f"{1e3 * img_fwhm:.3e}"))

        # update subap canvas
        if isinstance(subap_images, cp.ndarray):
            img = subap_images.get()
        else:
            img = np.asarray(subap_images)
        self.canvas_subap.set_image_masked(img, mask=ut.Pupil_tools.generate_pupil(img.shape[0]))

        self.busy_label.setText("Waiting")

    def recalculate(self, params):
        # request the calc worker to recompute (queued call)
        if hasattr(self, "calc_worker") and self.calc_worker is not None:
            QMetaObject.invokeMethod(self.calc_worker, "recompute", Qt.QueuedConnection)
        # also schedule an actuator update afterwards
        self._schedule_update(self.act_select_spinbox.value())

    def _on_spin_changed(self, v):
        self._schedule_update(v)

    def _schedule_update(self, v):
        self.pending_index = int(v)
        self.debounce_timer.start()

    def _emit_worker_request(self):
        self.job_id += 1
        jid = self.job_id
        k = int(self.pending_index)

        # now update_request is connected to calc_worker.process_subap, so this is queued
        self.busy_label.setText("Waiting")
        self.update_request.emit(k, jid)

    def closeEvent(self, event):
        # stop calc thread
        if hasattr(self, "calc_thread") and self.calc_thread is not None and self.calc_thread.isRunning():
            self.calc_thread.quit()
            self.calc_thread.wait(10)

        super().closeEvent(event)
