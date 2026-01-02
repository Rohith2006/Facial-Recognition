import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.face_router import router as face_router
from api.name_router import router as name_router
from api.image_router import router as image_router

app = FastAPI(
    title="Facial Recognition API",
    description="API for facial recognition tasks including detection, verification, and identification.",
    version="1.0.0"
)
# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(
    face_router,
    prefix="/face",
    tags=["face"]
)
app.include_router(
    image_router,
    prefix="/image",
    tags=["image"]
)

app.include_router(
    name_router,
    prefix="/name",
    tags=["name"]
)

# Basic root endpoints
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Facial Recognition API!"}

@app.get("/health")
async def health_check():
    return {"status": "OK"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)