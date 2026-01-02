import io
from fastapi import APIRouter
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from core.orchestrator import Orchestrator

router = APIRouter()
orchestrator = Orchestrator()

@router.get("/get_image/{face_id}")
async def get_image_by_face_id(face_id: str):
    try:
        image = orchestrator.image_db.get_image_by_face_id(face_id)
        if image is None:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)  # Important: reset pointer to beginning
        
        # Return as StreamingResponse with proper media type
        return StreamingResponse(img_byte_arr, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error retrieving image: {str(e)}")
