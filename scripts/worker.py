# scripts/worker.py
from PySide6.QtCore import QObject, QRunnable, Signal, Slot, QTimer, Qt
import scripts.utilities as ut
import cupy as cp
import time


# ---- Generic Fire and Forget worker 
class GenericWorkerSignals(QObject):
    finished = Signal(object)  # emit the created generator
    error = Signal(Exception)

class GenericWorker(QRunnable):
    def __init__(self, generator_func, **params):
        super().__init__()
        self.generator_func = generator_func
        self.params = params
        self.signals = GenericWorkerSignals()

    def run(self):
        try:
            gen, next_frame_func = self.generator_func(**self.params)
            self.signals.finished.emit((gen, next_frame_func))
        except Exception as e:
            self.signals.error.emit(e)

# Worker for turbulence screen viewers
class FrameWorker(QObject):
    frame_ready = Signal(object)

    def __init__(self, gen_factory, params, n_frames=None, delay=0.01):
        super().__init__()
        self.gen_factory = gen_factory    # phase screen generator callable: () -> (phase_obj, next_frame_callable)
        self.n_frames = n_frames
        self.params = params
        self.delay = delay
        self._running = False
        self._gen, self._next_frame = self.gen_factory(**self.params)

        self.step()

    @Slot()
    def _ensure_gen(self):
        # create generator (run inside worker thread)
        if self._gen is None:
            obj, next_fn = self.gen_factory(**self.params)
            self._gen = obj
            self._next_frame = next_fn

    @Slot()
    def step(self):
        try:
            self._ensure_gen()
            frame = self._next_frame()
            self.frame_ready.emit(frame)
        except Exception:
            import traceback
            traceback.print_exc()

    @Slot()
    def run(self):
        if self._running:
            return

        self._running = True
        self._ensure_gen()

        self._timer = QTimer()
        self._timer.setInterval( int(1000 / 30) )  # 30 fps
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    @Slot()
    def _tick(self):
        if not self._running:
            return
        frame = self._next_frame()
        self.frame_ready.emit(frame)

    @Slot()
    def stop(self):
        self._running = False

    @Slot()
    def reset(self):
        self.stop()
        self._gen = None
        self._next_frame = None
        self.step()



class CalculateWorker(QObject):
    initialized_result = Signal(object)         # dict result for initial/full calculation
    subap_finished = Signal(object, int)        # ( (centroids, sensor), job_id )
    finished = Signal()

    def __init__(self, params):
        super().__init__()
        self.params = params
        # placeholders for large arrays filled by initialize/recompute
        self.pupil = None
        self.act_centers = None
        self.influence_maps = None
        self.active_sub_aps = None
        self.sub_aps = None
        self.sub_aps_idx = None
        self.sub_ap_width = None
        self.sub_slice = None
        self.sub_pupils = None
        self.ref_centroids = None
        self.ref_images = None
        self.ref_science_image = None
        self.ref_strehl = None
        self.normalized_image = None
        self.science_image_plot = None
        self.n_act = 0

    @Slot()
    def initialize(self):
        """Initial full computation. Safe to call when moved to calc thread."""
        self._do_full_compute()
        res = {
            "pupil": self.pupil,
            "act_centers": self.act_centers,
            "influence_maps": self.influence_maps,
            "active_sub_aps": self.active_sub_aps,
            "sub_aps": self.sub_aps,
            "sub_aps_idx": self.sub_aps_idx,
            "sub_ap_width": self.sub_ap_width,
            "sub_slice": self.sub_slice,
            "sub_pupils": self.sub_pupils,
            "ref_centroids": self.ref_centroids,
            "ref_images": self.ref_images,
            "ref_science_image": self.ref_science_image,
            "ref_strehl": self.ref_strehl,
            "normalized_image": self.normalized_image,
            "science_image_plot": self.science_image_plot,
            "n_act": self.n_act,
            "ref_fwhm": float(self._compute_ref_fwhm()),
            "diff_lim": float(1.22 * self.params.get("science_lambda") / self.params.get("telescope_diameter"))
        }
        self.initialized_result.emit(res)

    @Slot()
    def recompute(self):
        """Recompute full data (called when params change)."""
        self._do_full_compute()
        res = {
            "pupil": self.pupil,
            "act_centers": self.act_centers,
            "influence_maps": self.influence_maps,
            "active_sub_aps": self.active_sub_aps,
            "sub_aps": self.sub_aps,
            "sub_aps_idx": self.sub_aps_idx,
            "sub_ap_width": self.sub_ap_width,
            "sub_slice": self.sub_slice,
            "sub_pupils": self.sub_pupils,
            "ref_centroids": self.ref_centroids,
            "ref_images": self.ref_images,
            "ref_science_image": self.ref_science_image,
            "ref_strehl": self.ref_strehl,
            "normalized_image": self.normalized_image,
            "science_image_plot": self.science_image_plot,
            "n_act": self.n_act,
            "ref_fwhm": float(self._compute_ref_fwhm()),
            "diff_lim": float(1.22 * self.params.get("science_lambda") / self.params.get("telescope_diameter"))
        }
        self.initialized_result.emit(res)

    def _do_full_compute(self):
        """Private: heavy compute (same as your previous calculate body)."""
        self.pupil = cp.asarray(ut.Pupil_tools.generate_pupil())
        self.act_centers = ut.Pupil_tools.generate_actuators(self.pupil)
        self.influence_maps = ut.Pupil_tools.generate_actuator_influence_map(self.act_centers, self.pupil)

        (self.active_sub_aps,
         self.sub_aps,
         self.sub_aps_idx,
         self.sub_ap_width,
         self.sub_slice,
         self.sub_pupils) = ut.Pupil_tools.generate_sub_apertures(self.pupil)

        self.ref_centroids, self.ref_images = ut.Analysis.generate_subaperture_images(
            0,
            pupil=self.pupil,
            influence_maps=[self.pupil],
            sub_aps_idx=self.sub_aps_idx,
            sub_ap_width=self.sub_ap_width,
            sub_pupils=self.sub_pupils,
        )

        self.ref_science_image, self.ref_strehl = ut.Analysis.generate_science_image(self.pupil, cp.zeros_like(self.pupil), pad=512)
        self.normalized_image = self.ref_science_image / self.ref_science_image.sum()
        self.science_image_plot = cp.log10(self.normalized_image[512:-512, 512:-512] + 1e-12)

        self.n_act = int(self.act_centers.shape[0])

    def _compute_ref_fwhm(self):
        plate_rad = (self.params.get("science_lambda") * self.params.get("grid_size")
                     / (self.params.get("telescope_diameter") * self.ref_science_image.shape[0]))
        return plate_rad * ut.Analysis.fwhm_radial(self.ref_science_image)

    @Slot(int, int)
    def process_subap(self, k, job_id):
        """Process per-actuator measurement in the same worker thread."""
        try:
            # reuse generate_subaperture_images which returns (centroids, sensor)
            centroids, sensor = ut.Analysis.generate_subaperture_images(
                k,
                pupil=self.pupil,
                influence_maps=self.influence_maps,
                sub_aps_idx=self.sub_aps_idx,
                sub_ap_width=self.sub_ap_width,
                sub_pupils=self.sub_pupils,
            )
            # Emit exactly like SubapWorker did: (centroids, sensor), job_id
            self.subap_finished.emit((centroids, sensor), job_id)
        except Exception:
            import traceback
            traceback.print_exc()
            # emit empty result to avoid hanging UI
            self.subap_finished.emit((cp.zeros((0,2)), cp.zeros((1,1))), job_id)
