from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl
import uvicorn
from typing import Optional, List
import os
import cv2
from google.generativeai import GenerativeModel
import google.generativeai as genai
from PIL import Image
import subprocess
from pathlib import Path
import uuid
import json

app = FastAPI(
    title="Video Frame Analyzer API",
    description="API for analyzing video frames using AI to detect specific objects",
    version="1.0.0"
)

@app.get("/")
async def root():
    """
    Root endpoint that provides API information and available endpoints.
    """
    return {
        "message": "Welcome to the Video Frame Analyzer API",
        "version": "1.0.0",
        "endpoints": {
            "POST /analyze": "Start a new video analysis",
            "GET /status/{task_id}": "Check the status of an analysis",
            "GET /results/{task_id}": "Get the results of a completed analysis"
        },
        "documentation": {
            "Swagger UI": "/docs",
            "ReDoc": "/redoc"
        }
    }

class AnalysisRequest(BaseModel):
    url: HttpUrl
    target_object: str
    fps: Optional[float] = 1
    batch_size: Optional[int] = 5

class AnalysisResponse(BaseModel):
    task_id: str
    status: str
    message: str

class AnalysisResult(BaseModel):
    timestamp: int
    frame: str
    description: str

# Store for background tasks
analysis_tasks = {}

class VideoAnalyzer:
    def __init__(self, target_object, task_id):
        self.setup_directories(task_id)
        self.target_object = target_object
        self.task_id = task_id
        self.api_key = "AIzaSyDoqwTDbzzjglJ6IKXq4KDDdEkZqfpvT-A"
        genai.configure(api_key=self.api_key)
        self.model = GenerativeModel("gemini-1.5-flash")
        
    def setup_directories(self, task_id):
        """Create task-specific directories"""
        self.base_dir = Path(f"tasks/{task_id}")
        self.downloads_dir = self.base_dir / "downloads"
        self.frames_dir = self.base_dir / "frames"
        
        # Create directories
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.downloads_dir.mkdir(exist_ok=True)
        self.frames_dir.mkdir(exist_ok=True)
        print(f"Created directories for task {task_id}")

    def download_video(self, url):
        """Download video using yt-dlp"""
        try:
            video_path = self.downloads_dir / "video.mp4"
            cmd = [
                "yt-dlp",
                "-f",
                "best",
                "-o",
                str(video_path),
                str(url)
            ]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return str(video_path) if video_path.exists() else None
        except Exception as e:
            print(f"Error downloading video: {e}")
            return None

    def extract_frames(self, video_path, fps=1):
        """Extract frames using ffmpeg"""
        try:
            output_pattern = str(self.frames_dir / "frame_%04d.jpg")
            cmd = [
                "ffmpeg",
                "-i",
                video_path,
                "-vf",
                f"fps={fps}",
                "-frame_pts",
                "1",
                output_pattern
            ]
            subprocess.run(cmd, check=True)
            return True
        except Exception as e:
            print(f"Error extracting frames: {e}")
            return False

    def analyze_frames(self, batch_size=5):
        """Analyze frames for the target object"""
        frames = sorted(self.frames_dir.glob("frame_*.jpg"))
        results = []

        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            batch_images = []
            
            try:
                for frame_path in batch_frames:
                    img = Image.open(frame_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    batch_images.append(img)

                for img, frame_path in zip(batch_images, batch_frames):
                    timestamp = int(frame_path.stem.split('_')[1])
                    
                    prompt = f"""Analyze this frame and determine if any {self.target_object} is present.
If {self.target_object} is present, describe:
1. Where the {self.target_object} appears in the frame
2. What type/variant of {self.target_object} it is (if applicable)
3. How prominently it is featured
Return ONLY 'No {self.target_object} detected' if no {self.target_object} is visible."""

                    try:
                        import time
                        time.sleep(4)
                        response = self.model.generate_content([prompt, img])
                        description = response.text if response.text else "No description generated"
                        
                        if description and f"No {self.target_object} detected" not in description:
                            results.append({
                                'timestamp': timestamp,
                                'frame': str(frame_path),
                                'description': description
                            })
                    except Exception as e:
                        print(f"Error processing frame {timestamp}: {e}")

            except Exception as e:
                print(f"Error analyzing batch starting at frame {i}: {e}")
                continue
            finally:
                for img in batch_images:
                    img.close()

        return results

async def process_video(task_id: str, url: str, target_object: str, fps: float, batch_size: int):
    """Background task to process video"""
    try:
        analysis_tasks[task_id]["status"] = "processing"
        
        analyzer = VideoAnalyzer(target_object, task_id)
        
        # Download video
        video_path = analyzer.download_video(url)
        if not video_path:
            analysis_tasks[task_id]["status"] = "failed"
            analysis_tasks[task_id]["message"] = "Failed to download video"
            return

        # Extract frames
        if not analyzer.extract_frames(video_path, fps):
            analysis_tasks[task_id]["status"] = "failed"
            analysis_tasks[task_id]["message"] = "Failed to extract frames"
            return

        # Analyze frames
        results = analyzer.analyze_frames(batch_size)
        
        # Save results
        results_file = Path(f"tasks/{task_id}/results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f)
        
        analysis_tasks[task_id]["status"] = "completed"
        analysis_tasks[task_id]["message"] = "Analysis completed successfully"
        analysis_tasks[task_id]["results"] = results

    except Exception as e:
        analysis_tasks[task_id]["status"] = "failed"
        analysis_tasks[task_id]["message"] = f"Analysis failed: {str(e)}"

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_video(request: AnalysisRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    analysis_tasks[task_id] = {
        "status": "queued",
        "message": "Task queued for processing"
    }
    
    background_tasks.add_task(
        process_video,
        task_id,
        str(request.url),
        request.target_object,
        request.fps,
        request.batch_size
    )
    
    return AnalysisResponse(
        task_id=task_id,
        status="queued",
        message="Analysis task has been queued"
    )

@app.get("/status/{task_id}", response_model=AnalysisResponse)
async def get_status(task_id: str):
    if task_id not in analysis_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = analysis_tasks[task_id]
    return AnalysisResponse(
        task_id=task_id,
        status=task["status"],
        message=task["message"]
    )

@app.get("/results/{task_id}")
async def get_results(task_id: str):
    if task_id not in analysis_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = analysis_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Task is not completed. Current status: {task['status']}")
    
    return task.get("results", [])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
