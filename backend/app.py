"""
Photogrammetry Backend API
Handles photo uploads and 3D model generation from multiple images.
"""
import os
import uuid
import json
import asyncio
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from photogrammetry import PhotogrammetryPipeline

app = FastAPI(title="Photogrammetry API", version="1.0.0")

# CORS for mobile access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = Path(__file__).parent.parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
FRONTEND_DIR = BASE_DIR / "frontend"

UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# In-memory job storage (use Redis/DB for production)
jobs = {}


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: int  # 0-100
    message: str
    created_at: str
    completed_at: Optional[str] = None
    model_url: Optional[str] = None
    preview_url: Optional[str] = None
    num_images: int = 0


class JobResponse(BaseModel):
    job_id: str
    message: str


@app.get("/")
async def root():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/api/upload", response_model=JobResponse)
async def upload_photos(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """Upload multiple photos for photogrammetry processing."""
    if len(files) < 3:
        raise HTTPException(
            status_code=400,
            detail="At least 3 photos required for 3D reconstruction"
        )

    if len(files) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 photos allowed per job"
        )

    # Create job
    job_id = str(uuid.uuid4())[:8]
    job_dir = UPLOADS_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    # Save uploaded files
    saved_files = []
    for i, file in enumerate(files):
        if not file.content_type or not file.content_type.startswith("image/"):
            continue

        ext = Path(file.filename).suffix.lower() or ".jpg"
        if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
            ext = ".jpg"

        file_path = job_dir / f"image_{i:03d}{ext}"
        content = await file.read()

        with open(file_path, "wb") as f:
            f.write(content)
        saved_files.append(str(file_path))

    if len(saved_files) < 3:
        raise HTTPException(
            status_code=400,
            detail="At least 3 valid image files required"
        )

    # Initialize job status
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0,
        "message": "Job queued for processing",
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "model_url": None,
        "preview_url": None,
        "num_images": len(saved_files),
        "files": saved_files,
    }

    # Start processing in background
    background_tasks.add_task(process_job, job_id)

    return JobResponse(
        job_id=job_id,
        message=f"Uploaded {len(saved_files)} images. Processing started."
    )


async def process_job(job_id: str):
    """Process photogrammetry job in background."""
    job = jobs.get(job_id)
    if not job:
        return

    try:
        job["status"] = "processing"
        job["message"] = "Initializing photogrammetry pipeline..."
        job["progress"] = 5

        output_dir = OUTPUTS_DIR / job_id
        output_dir.mkdir(exist_ok=True)

        # Progress callback
        def update_progress(progress: int, message: str):
            job["progress"] = progress
            job["message"] = message

        # Run photogrammetry pipeline
        pipeline = PhotogrammetryPipeline(
            image_paths=job["files"],
            output_dir=str(output_dir),
            progress_callback=update_progress
        )

        result = await asyncio.to_thread(pipeline.run)

        if result["success"]:
            job["status"] = "completed"
            job["progress"] = 100
            job["message"] = "3D model generated successfully!"
            job["model_url"] = f"/api/models/{job_id}/model.glb"
            job["preview_url"] = f"/api/models/{job_id}/preview.png"
            job["completed_at"] = datetime.now().isoformat()
        else:
            job["status"] = "failed"
            job["message"] = result.get("error", "Unknown error occurred")

    except Exception as e:
        job["status"] = "failed"
        job["message"] = f"Processing error: {str(e)}"


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a processing job."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatus(
        job_id=job["job_id"],
        status=job["status"],
        progress=job["progress"],
        message=job["message"],
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
        model_url=job.get("model_url"),
        preview_url=job.get("preview_url"),
        num_images=job.get("num_images", 0),
    )


@app.get("/api/models/{job_id}/{filename}")
async def get_model_file(job_id: str, filename: str):
    """Serve generated model files."""
    file_path = OUTPUTS_DIR / job_id / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Set correct content type
    content_types = {
        ".glb": "model/gltf-binary",
        ".gltf": "model/gltf+json",
        ".obj": "text/plain",
        ".mtl": "text/plain",
        ".ply": "application/octet-stream",
        ".png": "image/png",
        ".jpg": "image/jpeg",
    }

    ext = file_path.suffix.lower()
    content_type = content_types.get(ext, "application/octet-stream")

    return FileResponse(file_path, media_type=content_type)


@app.get("/api/jobs")
async def list_jobs():
    """List all jobs."""
    return [
        JobStatus(
            job_id=j["job_id"],
            status=j["status"],
            progress=j["progress"],
            message=j["message"],
            created_at=j["created_at"],
            completed_at=j.get("completed_at"),
            model_url=j.get("model_url"),
            preview_url=j.get("preview_url"),
            num_images=j.get("num_images", 0),
        )
        for j in jobs.values()
    ]


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its files."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    import shutil

    # Remove files
    upload_dir = UPLOADS_DIR / job_id
    output_dir = OUTPUTS_DIR / job_id

    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)

    del jobs[job_id]

    return {"message": "Job deleted"}


# Mount static files for frontend
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
