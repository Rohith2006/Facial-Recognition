import time
from face_engine.engine import FacialRecognitionEngine
import numpy as np
from PIL import Image
import onnxruntime as ort
print(ort.get_available_providers())

preload_start = time.time()
def test_face_engine_initialization():
    fr_engine = FacialRecognitionEngine(
            model_name="antelopev2",
            device=0,
            det_size=(320, 320)
        )
    
    img =  "1.jpg"  # Replace with actual image path
    image = Image.open(img).convert("RGB")
    embedding = fr_engine.get_embedding_from_pil(image)
    start_time = time.time()
    for i in range(1000):
        embedding = fr_engine.get_embedding_from_pil(image)
        print(f"Iteration {i}")
    # print(embedding)
    end_time = time.time()
    print(f"Time taken for 1000 iterations: {end_time - start_time} seconds")
    print(f"load time: {start_time - preload_start} seconds")
    print("FaceEngine initialization test passed.")

if __name__ == "__main__":
    test_face_engine_initialization()
