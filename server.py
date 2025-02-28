from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from moviepy.editor import VideoFileClip
import whisper
from transformers import pipeline
import os
import asyncio
import logging
from typing import Optional
from pydantic import BaseModel
import shutil
import tempfile
from supabase import create_client
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
import requests

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

env_path = Path("..") / ".env"
load_dotenv(dotenv_path=env_path)

class VideoProcessor:
    def __init__(self):
        # Initialising models
        self.whisper_model = whisper.load_model("base.en")
        self.qa_pipeline = pipeline("question-answering", model = "deepset/roberta-base-squad2")

        # Initialise Supabase client
        self.supabase = create_client(os.getenv("SUPABASE_URL"),
                                      os.getenv("SUPABASE_KEY"))

    async def process_video(self, file: UploadFile, user_id: str) -> dict:
        """
       Process video file asynchronously:
       1. Save to temp storage
       2. Extract audio
       3. Transcribe
       4. Store results in Supabase
       """
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                video_path = os.path.join(temp_dir, file.filename)
                logger.info(f"Saving file to {video_path}")

                with open(video_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)

                # Upload to Superbase storage
                logger.info(f"File saved, starting transcription")

                try:
                    with VideoFileClip(video_path) as video:
                        # Extract audio
                        audio_path = os.path.join(temp_dir, "audio.mp3")
                        video.audio.write_audiofile(audio_path, logger=None)

                        # Transcribe
                        logger.info("Starting transcription")
                        result = self.whisper_model.transcribe(audio_path)
                        transcription = result["text"]

                        # Upload to Supabase storage
                        logger.info("Uploading to Supabase")
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        storage_path = f"{user_id}/{timestamp}_{file.filename}"

                        with open(video_path, "rb") as fp:
                            try:
                                self.supabase.storage.from_("videos").upload(
                                    storage_path,
                                    fp
                                )
                                video_url = self.supabase.storage.from_("videos").get_public_url(storage_path)
                            except Exception as e:
                                logger.error(f"Supabase upload failed: {str(e)}")
                                video_url = None

                        # Store in database
                        data = {
                            "user_id": user_id,
                            "filename": file.filename,
                            "video_url": video_url,
                            "transcription": transcription,
                            "created_at": datetime.now().isoformat()
                        }

                        self.supabase.table("transcriptions").insert(data).execute()

                        return {
                            "status": "success",
                            "transcription": transcription,
                            "video_url": video_url
                        }

                except Exception as e:
                    logger.error(f"Video processing error: {str(e)}")
                    raise

        except Exception as e:
            logger.exception(e)

    async def _upload_to_storage(self, file_path: str, filename: str, user_id: str) -> str:
        """Upload file to Superbase storage and return URL"""
        try:
            timestamp = datetime.now().strftime("%")
            storage_path = f"{user_id}/{timestamp}_{filename}"

            with open(file_path, "rb") as fp:
                self.supabase.storage.from_("videos").upload(storage_path, fp)

            return self.supabase.storage.from_("videos").get_public_url(storage_path)

        except Exception as e:
            logger.exception(e)

    async def _transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio file using whisper"""
        try:
            result = self.whisper_model.transcribe(audio_path)
            return result["text"]
        except Exception as e:
            logger.exception(e)

    async def _store_transcription(self, filename: str, transcription: str, video_url: str, user_id: str):
        """Store transcription metadata in Superbase database"""
        try:
            data = {"user_id": user_id,
                    "filename": filename,
                    "video_url": video_url,
                    "transcription": transcription,
                    "created_at": datetime.now().isoformat()}

            self.supabase.table("transcriptions").insert(data).execute()
            logger.info("Transcription stored in transcriptions table ")

        except Exception as e:
            logger.exception(e)


# Initialise app with CORS
app = FastAPI()
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],  # Update this with frontend URL in production
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])

# Initialise video processor
processor = VideoProcessor()


@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...),
                       background_tasks: BackgroundTasks = None,
                       user_id: Optional[str] = None):

    """Endpoint to handle video upload and processing"""
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID required")

    try:
        result = await processor.process_video(file, user_id)
        return result
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/transcriptions/{user_id}")
async def get_transcriptions(user_id: str):
    """Retrieve all transcriptions for a user"""
    try:
        result = processor.supabase.table("transcriptions").select("*").eq("user_id", user_id).execute()
        return result.data
    except Exception as e:
        logger.exception(e)
        logger.("There was an error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
