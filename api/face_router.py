from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from core.orchestrator import Orchestrator
from utils.logger import get_logger
from PIL import Image
import io

router = APIRouter()
orchestrator = Orchestrator()
logger = get_logger()

@router.post("/identify")
async def identify_face(file: UploadFile = File(...)):
    try:
        contents = await file.read()        
        image = Image.open(io.BytesIO(contents))
        response = orchestrator.identify(image)
        logger.info(f"Identify response: {response}")
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in identify_face: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@router.post("/register")
async def register_face(name: str, file: UploadFile = File(...)):
    try:
        contents = await file.read()        
        image = Image.open(io.BytesIO(contents))
        face_id = orchestrator.register(image, name)
        logger.info(f"Registered face ID: {face_id} for name: {name}")
        return {"response": face_id}
    except Exception as e:
        logger.error(f"Error in register_face: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
    