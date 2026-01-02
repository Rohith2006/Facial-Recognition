import threading
from insightface.app import FaceAnalysis
import onnxruntime as ort
import os
import torch
import numpy as np
import cv2
from PIL import Image
import warnings
import sys

warnings.filterwarnings("ignore", category=FutureWarning)
os.makedirs('./trt_engine_cache', exist_ok=True)
ort.set_default_logger_severity(3)

def pil_to_bgr(image: Image.Image) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")
    img = np.asarray(image)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

class DummyFile:
    def write(self, x): pass
    def flush(self): pass

class FacialRecognitionEngine:
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        device=None,
        det_size=(320, 320),
        model_name="antelopev2"
    ):
        # Prevent re-initialization
        if self.__class__._initialized:
            return

        self.__class__._initialized = True

        # Silence InsightFace stdout
        sys.stdout = DummyFile()

        # TensorRT config (keep disabled unless TRT EP is enabled)
        trt_options = {
            "trt_builder_optimization_level": 1,
            "trt_engine_cache_path": "./trt_engine_cache",
            "trt_max_workspace_size": 1 << 30,
            "trt_fp16_enable": True,
            "trt_context_memory_sharing_enable": True,
        }

        providers = [
            # ("TensorrtExecutionProvider", trt_options),
            ("CUDAExecutionProvider", {}),
            ("CPUExecutionProvider", {}),
        ]

        self.device = device if device is not None else (0 if self._cuda_available() else -1)
        self.det_size = det_size

        print(
            f"Using device: {'GPU' if self.device >= 0 else 'CPU'} "
            f"(ctx_id={self.device}), det_size={det_size}, model={model_name}"
        )

        self.app = FaceAnalysis(
            name=model_name,
            root="./insightface_models",
            providers=providers,
        )

        self.app.prepare(
            ctx_id=self.device,
            det_size=det_size,
            det_thresh=0.3,
        )

        sys.stdout = sys.__stdout__

    def _cuda_available(self) -> bool:
        try:
            return torch.cuda.is_available()
        except Exception:
            return False

    def get_embedding_from_pil(self, img: Image.Image) -> np.ndarray:
        img_array = np.array(img)

        if img_array.shape[-1] == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array

        faces = self.app.get(img_bgr)
        if not faces:
            raise ValueError("No face detected in image")

        return faces[0].embedding
    
    # def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
    #     """Returns cosine similarity between two embeddings.
        
    #     Note: InsightFace embeddings are already L2-normalized,
    #     so cosine similarity = dot product.
    #     """
    #     return float(np.dot(emb1, emb2))
    
    # def _l2_normalize(self, emb: np.ndarray) -> np.ndarray:
    #     """Return L2-normalized copy of embedding.
        
    #     Note: InsightFace embeddings are already normalized.
    #     """
    #     n = norm(emb)
    #     if n == 0:
    #         return emb
    #     return emb / (n + 1e-8)
    
    # def compare_faces(self, img1_path: str, img2_path: str, threshold: float = 0.3):
    #     """Returns if two images are same face with similarity score.
        
    #     Note: InsightFace typical thresholds:
    #     - 0.4: High precision (recommended)
    #     - 0.3: Balanced
    #     - 0.25: High recall
    #     """
    #     emb1 = self.get_embedding(img1_path)
    #     emb2 = self.get_embedding(img2_path)
        
    #     # Cosine similarity (dot product for normalized vectors)
    #     similarity = self.cosine_similarity(emb1, emb2)
        
    #     # Calculate angle in degrees
    #     c = max(-1.0, min(1.0, similarity))
    #     angle_deg = float(np.degrees(np.arccos(c)))
        
    #     result = {
    #         "similarity": similarity,
    #         "angle_degrees": angle_deg,
    #         "same_person": similarity > threshold,
    #     }
        
    #     return result
        

    
