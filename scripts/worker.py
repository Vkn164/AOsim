# scripts/worker.py
from PySide6.QtCore import QObject, Signal, Slot
from scripts.utilities import Analysis
import cupy as cp


class SubapWorker(QObject):
    finished = Signal(object, int)  # (result, job_id)

    def __init__(
        self,
        pupil,
        influence_maps,
        sub_aps_idx,
        sub_ap_width,
        sub_pupils
    ):
        super().__init__()
        self.pupil = pupil
        self.influence_maps = influence_maps
        self.sub_aps_idx = sub_aps_idx
        self.sub_ap_width = sub_ap_width
        self.sub_pupils = sub_pupils

    @Slot(int, int)
    def process(self, k, job_id):
        result = Analysis.generate_subaperture_images(
            int(k),
            pupil=self.pupil,
            influence_maps=self.influence_maps,
            sub_aps_idx=self.sub_aps_idx,
            sub_ap_width=self.sub_ap_width,
            sub_pupils=self.sub_pupils,
        )
        self.finished.emit(result, job_id)
