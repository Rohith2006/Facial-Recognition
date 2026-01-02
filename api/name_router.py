from fastapi import APIRouter
from fastapi import HTTPException
from core.orchestrator import Orchestrator
from utils.logger import get_logger

logger = get_logger()
router = APIRouter()
orchestrator = Orchestrator()

@router.get("/unnamed")
async def get_unnamed_faces():
    try:
        unnamed_faces = orchestrator.get_unnamed_faces()
        logger.info(f"Retrieved unnamed faces: {unnamed_faces}")
        return {"unnamed_faces": unnamed_faces}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error retrieving unnamed faces: {str(e)}")
    
@router.get("/get_name/{face_id}")
async def get_name_by_face_id(face_id: str):
    try:
        result = orchestrator.image_db.retrieve_by_face_id(face_id)
        if result is None:
            logger.warning(f"Face ID not found: {face_id}")
            raise HTTPException(status_code=404, detail="Face ID not found")
        face_id, name = result
        logger.info(f"Retrieved name: {name} for face_id: {face_id}")
        return {"face_id": face_id, "name": name}
    except Exception as e:
        logger.error(f"Error retrieving name for face_id {face_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error retrieving name: {str(e)}")
    
@router.put("/update_name/{face_id}")
async def update_name_by_face_id(face_id: str, name: str):
    try:
        result = orchestrator.image_db.update_name(face_id, name)
        if not result:
            logger.warning(f"Face ID not found for update: {face_id}")
            raise HTTPException(status_code=404, detail="Face ID not found")
        logger.info(f"Updated name to: {name} for face_id: {face_id}")
        return {"status": "success", "face_id": face_id, "name": name}
    except Exception as e:
        logger.error(f"Error updating name for face_id {face_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error updating name: {str(e)}")