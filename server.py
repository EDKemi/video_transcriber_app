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

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoProcessor:
    def __init__(self):
        # Initialising models
        self.whisper_model = whisper.load_model("medium.en")
        self.qa_pipeline = pipeline("question-answering", model = "deepset/roberta-base-squad2")

        # Initialise Supabase client
        self.supabase = create_client(os.getenv("SUPABASE_URL"),
                                      os.getenv("SUPABASE_KEY"))

    async def process_video(self, file: UploadFile, user_id: str) -> dict:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                video_path = os.path.join(temp_dir, file.filename)
                with open(video_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)

            # upload to Superbase storage
            video_url = await self._upload_to_storage(video_path, file.file, user_id)

            # Extract the audio in the background
            audio_path = os.path.join(temp_dir, "audio.mp3")
            # video = VideoFileClip(video_path)
            # video.audio.write_audiofile(audio_path, logger=None)
            # video.close()

            with VideoFileClip(video_path) as video:
                video.audio.write_audiofile(audio_path, logger=None)

            # Transcribe audio
            transcription = await self._transcribe_audio(audio_path)

            # Store transcription in database
            await self._store_transcrition(file.filename, transcription, video_url)

        except Exception as e:
            logger.exception(e)




