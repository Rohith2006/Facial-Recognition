from face_engine.engine import FacialRecognitionEngine
from core.quality_check import QualityCheck
from vector_db.store import VectorStore
from storage.db import ImageDB
from PIL import Image
import numpy as np
import uuid as uuid
from uuid import uuid4
from utils.logger import get_logger

logger = get_logger()

class Orchestrator:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Orchestrator, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_name: str = "antelopev2",
        device: int = None,
        det_size: tuple = (320, 320),
        similarity_threshold: float = 0.3
    ):
        if self.__class__._initialized:
            return

        self.fr_engine = FacialRecognitionEngine(
            model_name=model_name,
            device=device,
            det_size=det_size
        )
        self.similarity_threshold = similarity_threshold
        self.quality_checker = QualityCheck()
        self.vector_store = VectorStore(dim=512)
        self.image_db = ImageDB()

        logger.info("Orchestrator initialized (singleton).")

        self.__class__._initialized = True
    
    def quality_check(self, image: Image.Image) -> bool:
        """Perform basic quality checks on the image."""
        # size of face is at least 160x160
        if not self.quality_checker.min_face_size(image, min_size=80):
            logger.warning("Face too small")
            return False
        
        
        # face is not fromtal +-20 degrees is acceptable
        # if not self.quality_checker.is_frontal(image, angle_threshold=20):
        #     logger.warning("Face not frontal")
        #     return "Face not frontal"

        return True
    
    # identify method - just convert face_id to string when storing/retrieving
    def identify(self, image: Image.Image):

        quality_passed = self.quality_check(image)
        if not quality_passed:
            return {"quality_check": "failed", "reason": "Face too small"}

        embedding = self.fr_engine.get_embedding_from_pil(image)
        response = self.vector_store.search(embedding, top_k=1)
        
        if not response:
            response = [(None, 0.0)]  # No match found
        logger.info(f"Identification response: {response}")
        
        if response[0][1] >= self.similarity_threshold:
            print(f"Match found with similarity: {response[0][1]}")
            face_id = str(response[0][0])  # Convert to string
            face_id, name = self.image_db.retrieve_by_face_id(face_id)
            logger.info(f"Matched face_id: {face_id} with name: {name}")
            return {"type" : "matched", "face_id": face_id, "name": name, "similarity": str(response[0][1])}
        
        else:
            face_id = self.vector_store.add_embedding(embedding)
            face_id = str(face_id)  # Convert to string
            logger.info(f"New embedding added with index: {face_id}")
            self.image_db.store_image(image, face_id=face_id)
            face_id, name = self.image_db.retrieve_by_face_id(face_id)
            logger.info(f"Registered new face_id: {face_id} with name: {name}")
        return {"type" : "registered", "face_id": face_id, "name": name, "similarity": "N/A"}  
    
    
    def register(self, image: Image.Image, name: str):
        if not self.quality_check(image):
            return {"status": "failed", "reason": "Image failed quality checks"}

        embedding = self.fr_engine.get_embedding_from_pil(image)
        response = self.vector_store.search(embedding, top_k=1)
        if response and response[0][1] >= self.similarity_threshold:
            face_id = str(response[0][0])  # Convert to string
            logger.info(f"Face already registered with similarity: {response[0][1]}")
            self.image_db.store_image(image, face_id=face_id, name=name)
            return {"status": "exists", "face_id": face_id, "name": name}
        # If no match found, register new embedding
        face_id = self.vector_store.add_embedding(embedding)
        face_id = str(face_id)  # Convert to string
        logger.info(f"New embedding added with index: {face_id}")
        self.image_db.store_image(image, face_id=face_id, name=name)
        return {"status": "success", "face_id": face_id, "name": name}
    
    def register_with_id(self, id: str, name: str):
        face_id = str(id)
        self.image_db.update_name(face_id, name)
        logger.info(f"Updated name to: {name} for face_id: {face_id}")
        return {"status": "success", "face_id": face_id, "name": name}
    
    def get_unnamed_faces(self):
        unnamed_faces = self.image_db.get_unnamed_faces()
        logger.info(f"Retrieved unnamed faces: {unnamed_faces}")
        return unnamed_faces
    
    
