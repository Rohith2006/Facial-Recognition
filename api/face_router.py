from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from core.orchestrator import Orchestrator
from PIL import Image
import io

router = APIRouter()
orchestrator = Orchestrator()

@router.post("/identify")
async def identify_face(file: UploadFile = File(...)):
    try:
        contents = await file.read()        
        image = Image.open(io.BytesIO(contents))
        response = orchestrator.identify(image)
        
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@router.post("/register")
async def register_face(name: str, file: UploadFile = File(...)):
    try:
        contents = await file.read()        
        image = Image.open(io.BytesIO(contents))
        face_id = orchestrator.register(image, name)
        
        return {"response": face_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
    