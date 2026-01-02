from fastapi import APIRouter
from fastapi import HTTPException
from core.orchestrator import Orchestrator

router = APIRouter()
orchestrator = Orchestrator()

@router.get("/unnamed")
async def get_unnamed_faces():
    try:
        unnamed_faces = orchestrator.get_unnamed_faces()
        return {"unnamed_faces": unnamed_faces}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error retrieving unnamed faces: {str(e)}")
    
@router.get("/get_name/{face_id}")
async def get_name_by_face_id(face_id: str):
    try:
        result = orchestrator.image_db.retrieve_by_face_id(face_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Face ID not found")
        face_id, name = result
        return {"face_id": face_id, "name": name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error retrieving name: {str(e)}")
    
@router.put("/update_name/{face_id}")
async def update_name_by_face_id(face_id: str, name: str):
    try:
        result = orchestrator.image_db.update_name(face_id, name)
        return {"status": "success", "face_id": face_id, "name": name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error updating name: {str(e)}")