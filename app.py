from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import shutil
import uuid

from pipeline import run_cctv_pipeline

app = FastAPI(title="CCTV Incident Pipeline API", version="1.0")

BASE_UPLOADS = Path("uploads")
BASE_RUNS = Path("runs")
BASE_UPLOADS.mkdir(parents=True, exist_ok=True)
BASE_RUNS.mkdir(parents=True, exist_ok=True)


@app.post("/process")
async def process_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="Unsupported video format")

    run_id = f"run_{uuid.uuid4().hex}"
    run_dir = BASE_RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    input_path = run_dir / file.filename
    with input_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Run pipeline
    artifacts = run_cctv_pipeline(
        video_path=str(input_path),
        run_id=run_id,
        run_dir=str(run_dir),
    )

    # Return JSON containing artifact paths (frontend can download them)
    return JSONResponse({
        "run_id": run_id,
        "artifacts": artifacts
    })


@app.get("/download")
def download(path: str):
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(p), filename=p.name)
