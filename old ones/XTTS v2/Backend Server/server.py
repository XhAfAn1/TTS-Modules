import os
import uuid
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

# ------------------------------
# FIX FOR PYTORCH 2.6+ 
# ------------------------------
_original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

# ------------------------------
# DEVICE SETUP
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# FASTAPI APP
# ------------------------------
app = FastAPI(title="XTTS v2 TTS API")

# ------------------------------
# LOAD XTTS MODEL
# ------------------------------
from TTS.api import TTS

try:
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print(f"[SUCCESS] XTTS v2 model loaded on {device}")
except Exception as e:
    print("[ERROR] Failed to load TTS model:", e)
    raise e

# ------------------------------
# DEFAULT SPEAKER REFERENCE
# ------------------------------
DEFAULT_SPEAKER_WAV = "test.wav"  # Change this to your audio file name

if not os.path.exists(DEFAULT_SPEAKER_WAV):
    print(f"[WARNING] Default speaker file '{DEFAULT_SPEAKER_WAV}' not found!")
else:
    print(f"[SUCCESS] Using default speaker: {DEFAULT_SPEAKER_WAV}")

# ------------------------------
# REQUEST MODEL
# ------------------------------
class TTSRequest(BaseModel):
    text: str
    speaker_wav: Optional[str] = None
    language: str = "en"
    
    # Voice control parameters
    speed: Optional[float] = 1.0  # 0.5 to 2.0 (slower to faster)
    temperature: Optional[float] = 0.75  # 0.1 to 1.0 (more consistent to more varied)
    length_penalty: Optional[float] = 1.0  # Controls speech length
    repetition_penalty: Optional[float] = 5.0  # Reduces repetition
    top_k: Optional[int] = 50  # Limits vocabulary (lower = more focused)
    top_p: Optional[float] = 0.85  # Nucleus sampling (lower = more consistent)
    
    # Emotion hints (add these to your text for better results)
    # Examples: "**excited**", "**sad**", "**angry**", "**whisper**"

class EmotionPreset(BaseModel):
    """Predefined emotion presets"""
    text: str
    emotion: str  # happy, sad, angry, excited, calm, whisper
    speaker_wav: Optional[str] = None
    language: str = "en"

# ------------------------------
# EMOTION PRESETS
# ------------------------------
EMOTION_SETTINGS = {
    "happy": {
        "temperature": 0.85,
        "speed": 1.1,
        "top_p": 0.9,
        "prefix": "**excited** ",
    },
    "sad": {
        "temperature": 0.65,
        "speed": 0.9,
        "top_p": 0.75,
        "prefix": "**sadly** ",
    },
    "angry": {
        "temperature": 0.95,
        "speed": 1.15,
        "top_p": 0.95,
        "prefix": "**angrily** ",
    },
    "excited": {
        "temperature": 0.95,
        "speed": 1.2,
        "top_p": 0.95,
        "prefix": "**very excited** ",
    },
    "calm": {
        "temperature": 0.6,
        "speed": 0.95,
        "top_p": 0.7,
        "prefix": "**calmly** ",
    },
    "whisper": {
        "temperature": 0.5,
        "speed": 0.85,
        "top_p": 0.6,
        "prefix": "**whispers** ",
    },
}

# ------------------------------
# ENDPOINTS
# ------------------------------
@app.get("/")
async def root():
    return {
        "message": "XTTS v2 TTS API is running!", 
        "device": device,
        "default_speaker": DEFAULT_SPEAKER_WAV,
        "available_emotions": list(EMOTION_SETTINGS.keys()),
        "voice_controls": {
            "speed": "0.5 to 2.0 (default: 1.0)",
            "temperature": "0.1 to 1.0 (default: 0.75)",
            "top_p": "0.1 to 1.0 (default: 0.85)"
        }
    }

@app.post("/speak")
async def speak(req: TTSRequest):
    """
    Generate speech with custom voice settings.
    
    Parameters:
    - speed: 0.5 (slower) to 2.0 (faster)
    - temperature: 0.1 (consistent) to 1.0 (varied/emotional)
    - top_p: 0.1 (focused) to 1.0 (creative)
    """
    try:
        os.makedirs("output", exist_ok=True)
        filename = f"output/{uuid.uuid4().hex}.wav"
        
        speaker_file = req.speaker_wav if req.speaker_wav else DEFAULT_SPEAKER_WAV
        
        if not os.path.exists(speaker_file):
            raise HTTPException(
                status_code=400,
                detail=f"Speaker file not found: {speaker_file}"
            )
        
        print(f"[INFO] Generating with settings: speed={req.speed}, temp={req.temperature}, top_p={req.top_p}")
        
        # Generate speech with custom parameters
        tts.tts_to_file(
            text=req.text,
            speaker_wav=speaker_file,
            language=req.language,
            file_path=filename,
            speed=req.speed,
            temperature=req.temperature,
            length_penalty=req.length_penalty,
            repetition_penalty=req.repetition_penalty,
            top_k=req.top_k,
            top_p=req.top_p,
        )
        
        print(f"[SUCCESS] Generated: {filename}")
        
        return FileResponse(
            path=filename, 
            filename=os.path.basename(filename), 
            media_type="audio/wav"
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/speak_emotion")
async def speak_with_emotion(req: EmotionPreset):
    """
    Generate speech with emotion presets.
    
    Available emotions: happy, sad, angry, excited, calm, whisper
    """
    try:
        if req.emotion not in EMOTION_SETTINGS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown emotion. Available: {list(EMOTION_SETTINGS.keys())}"
            )
        
        os.makedirs("output", exist_ok=True)
        filename = f"output/{uuid.uuid4().hex}.wav"
        
        speaker_file = req.speaker_wav if req.speaker_wav else DEFAULT_SPEAKER_WAV
        
        if not os.path.exists(speaker_file):
            raise HTTPException(
                status_code=400,
                detail=f"Speaker file not found: {speaker_file}"
            )
        
        # Get emotion settings
        emotion_config = EMOTION_SETTINGS[req.emotion]
        
        # Add emotion prefix to text
        modified_text = emotion_config["prefix"] + req.text
        
        print(f"[INFO] Generating with emotion: {req.emotion}")
        
        # Generate speech with emotion settings
        tts.tts_to_file(
            text=modified_text,
            speaker_wav=speaker_file,
            language=req.language,
            file_path=filename,
            speed=emotion_config["speed"],
            temperature=emotion_config["temperature"],
            top_p=emotion_config["top_p"],
        )
        
        print(f"[SUCCESS] Generated: {filename}")
        
        return FileResponse(
            path=filename, 
            filename=os.path.basename(filename), 
            media_type="audio/wav"
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/emotions")
async def list_emotions():
    """List available emotion presets and their settings"""
    return {
        "available_emotions": list(EMOTION_SETTINGS.keys()),
        "settings": EMOTION_SETTINGS
    }