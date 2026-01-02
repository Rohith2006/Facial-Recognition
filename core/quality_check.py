import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image


class QualityCheck:
    def __init__(self):
        self.app = FaceAnalysis(
            name="antelopev2",
            root="./insightface_models",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(320, 320))

    def _get_largest_face(self, image: Image.Image):
        faces = self.app.get(np.asarray(image.convert("RGB")))
        if not faces:
            return None
        return max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )

    # ----------------------------
    # Face size check
    # ----------------------------
    def min_face_size(self, image: Image.Image, min_size: int = 160) -> bool:
        face = self._get_largest_face(image)
        if face is None:
            return False

        w = face.bbox[2] - face.bbox[0]
        h = face.bbox[3] - face.bbox[1]

        return w >= min_size and h >= min_size

    # ----------------------------
    # Frontal check (≈ ±20°)
    # ----------------------------
    def is_frontal(self, image: Image.Image, angle_threshold: float = 20) -> bool:
        face = self._get_largest_face(image)
        if face is None:
            return False

        kps = face.kps
        le, re, nose, lm, rm = kps

        w = face.bbox[2] - face.bbox[0]
        h = face.bbox[3] - face.bbox[1]

        # ---- Roll (eye alignment) ----
        eye_dx = re[0] - le[0]
        eye_dy = abs(re[1] - le[1])
        if eye_dy / eye_dx > 0.15:
            return False

        # ---- Yaw (nose horizontal offset) ----
        eye_mid_x = (le[0] + re[0]) / 2
        yaw_ratio = abs(nose[0] - eye_mid_x) / w
        if yaw_ratio > 0.08:  # ≈ ±20°
            return False

        # ---- Pitch (nose–mouth vertical ratio) ----
        mouth_mid_y = (lm[1] + rm[1]) / 2
        pitch_ratio = abs(nose[1] - mouth_mid_y) / h
        if pitch_ratio < 0.18 or pitch_ratio > 0.35:
            return False

        return True
