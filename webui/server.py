import os
import time
import json
import asyncio
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engines import ENGINE_REGISTRY, ENGINE_METADATA, ENGINE_VOICES, BaseTTSEngine, SynthesisResult

app = FastAPI(title="TTS Module Comparison", version="1.0.0")

# Directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "samples")
PRESETS_FILE = os.path.join(BASE_DIR, "data", "presets", "aac_phrases.json")
STATIC_DIR = os.path.join(BASE_DIR, "webui", "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "webui", "templates")

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Static mounts
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

def ensure_wav_format(file_path: str) -> str:
    """Ensures audio reference file is converted to a clean 24kHz mono WAV for cloning."""
    if not file_path or not os.path.exists(file_path):
        return file_path
    if file_path.lower().endswith(".wav"):
        return file_path

    base, _ = os.path.splitext(file_path)
    wav_path = f"{base}.wav"
    try:
        import av
        import soundfile as sf
        import numpy as np
        import scipy.signal as sps
        container = av.open(file_path)
        stream = container.streams.audio[0]
        frames = []
        for frame in container.decode(stream):
            frames.extend(frame.to_ndarray())
        audio = np.concatenate(frames, axis=-1)
        sr = stream.rate
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        if sr != 24000:
            num_target = int(len(audio) * 24000 / sr)
            audio = sps.resample(audio, num_target)
            sr = 24000
        sf.write(wav_path, audio, sr)
        print(f"[Audio Converter] Converted '{file_path}' -> '{wav_path}' (24kHz mono)")
        return wav_path
    except Exception as e:
        print(f"[Audio Converter] Conversion failed for {file_path}: {e}")
        return file_path

# Engine instances cache
import gc
import torch

loaded_engines: Dict[str, BaseTTSEngine] = {}
HEAVY_GPU_ENGINES = {"vibevoice_realtime", "chatterbox_multilingual", "chatterbox_turbo", "cosyvoice"}

def get_or_load_engine(engine_id: str) -> BaseTTSEngine:
    # VRAM Safety Guard for 8GB GPU: Offload other heavy models before loading a new heavy model
    if engine_id in HEAVY_GPU_ENGINES:
        for old_id in list(loaded_engines.keys()):
            if old_id in HEAVY_GPU_ENGINES and old_id != engine_id:
                print(f"[VRAM Guard] Offloading {old_id} to keep GPU memory safe under 4GB...")
                del loaded_engines[old_id]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if engine_id not in loaded_engines:
        if engine_id not in ENGINE_REGISTRY:
            raise HTTPException(status_code=404, detail=f"Engine '{engine_id}' not found.")
        engine_cls = ENGINE_REGISTRY[engine_id]
        instance = engine_cls()
        loaded_engines[engine_id] = instance
    return loaded_engines[engine_id]


# Request models
class SynthesizeRequest(BaseModel):
    engine_id: str
    text: str
    voice: Optional[str] = None
    gender: Optional[str] = "female"
    speed: float = 1.0
    reference_audio: Optional[str] = None

class BatchSynthesizeRequest(BaseModel):
    engine_ids: List[str]
    text: str
    gender: Optional[str] = "female"
    speed: float = 1.0
    reference_audio: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/engines")
async def list_engines():
    result = []
    for meta in ENGINE_METADATA:
        eid = meta["id"]
        is_active = (eid in loaded_engines and loaded_engines[eid].is_loaded)
        voices = ENGINE_VOICES.get(eid, [])
        item = dict(meta)
        item["is_loaded"] = is_active
        item["voices"] = voices
        result.append(item)
    return {"engines": result}


@app.get("/api/presets")
async def get_presets():
    if os.path.exists(PRESETS_FILE):
        with open(PRESETS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    return {"categories": []}


@app.get("/api/samples")
async def list_samples():
    files = []
    if os.path.exists(SAMPLES_DIR):
        for fn in os.listdir(SAMPLES_DIR):
            if fn.lower().endswith((".wav", ".mp3", ".ogg", ".m4a")):
                files.append({
                    "filename": fn,
                    "url": f"/samples/{fn}",
                    "path": os.path.join(SAMPLES_DIR, fn)
                })
    return {"samples": files}


@app.post("/api/upload-sample")
async def upload_sample(file: UploadFile = File(...)):
    filename = file.filename
    clean_name = os.path.basename(filename)
    target_path = os.path.join(SAMPLES_DIR, clean_name)
    with open(target_path, "wb") as f:
        content = await file.read()
        f.write(content)

    final_path = ensure_wav_format(target_path)
    final_name = os.path.basename(final_path)
    return {
        "status": "success",
        "filename": final_name,
        "url": f"/samples/{final_name}",
        "path": final_path
    }


@app.post("/api/synthesize")
async def synthesize_single(req: SynthesizeRequest):
    engine = get_or_load_engine(req.engine_id)
    voice_to_use = req.voice
    if not voice_to_use and req.gender:
        voice_to_use = req.gender

    ref_to_use = req.reference_audio
    if req.gender == "male":
        male_sample = os.path.join(SAMPLES_DIR, "male_reference.wav")
        if (not ref_to_use or "default_ref.wav" in str(ref_to_use)) and os.path.exists(male_sample):
            ref_to_use = male_sample

    if ref_to_use and os.path.exists(ref_to_use):
        ref_to_use = ensure_wav_format(ref_to_use)

    # Run in thread pool to not block async event loop
    loop = asyncio.get_event_loop()
    result: SynthesisResult = await loop.run_in_executor(
        None,
        lambda: engine.synthesize(
            text=req.text,
            voice=voice_to_use,
            speed=req.speed,
            reference_audio=ref_to_use,
            output_dir=OUTPUTS_DIR
        )
    )
    return result


@app.post("/api/synthesize-batch")
async def synthesize_batch(req: BatchSynthesizeRequest):
    loop = asyncio.get_event_loop()
    results = []

    voice_to_use = req.gender or "female"
    ref_to_use = req.reference_audio
    if req.gender == "male":
        male_sample = os.path.join(SAMPLES_DIR, "male_reference.wav")
        if (not ref_to_use or "default_ref.wav" in str(ref_to_use)) and os.path.exists(male_sample):
            ref_to_use = male_sample

    if ref_to_use and os.path.exists(ref_to_use):
        ref_to_use = ensure_wav_format(ref_to_use)

    for eid in req.engine_ids:
        try:
            engine = get_or_load_engine(eid)
            res = await loop.run_in_executor(
                None,
                lambda e=engine: e.synthesize(
                    text=req.text,
                    voice=voice_to_use,
                    speed=req.speed,
                    reference_audio=ref_to_use,
                    output_dir=OUTPUTS_DIR
                )
            )
            results.append(res)
            if eid in HEAVY_GPU_ENGINES:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception as e:
            results.append({
                "engine_id": eid,
                "engine_name": eid,
                "text": req.text,
                "audio_path": "",
                "audio_url": "",
                "sample_rate": 0,
                "audio_duration_s": 0.0,
                "latency_ms": 0.0,
                "real_time_factor": 0.0,
                "status": "error",
                "error": str(e)
            })

    return {"results": results}


@app.post("/api/benchmark-category")
async def benchmark_category(category_name: str, engine_ids: List[str]):
    # Read category phrases
    if not os.path.exists(PRESETS_FILE):
        raise HTTPException(status_code=400, detail="Presets file missing.")
    with open(PRESETS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    selected_phrases = []
    for cat in data.get("categories", []):
        if cat["name"] == category_name or category_name == "ALL":
            selected_phrases.extend(cat["phrases"])

    if not selected_phrases:
        raise HTTPException(status_code=404, detail="Category not found or empty.")

    loop = asyncio.get_event_loop()
    summary = {}

    for eid in engine_ids:
        engine = get_or_load_engine(eid)
        latencies = []
        rtfs = []
        durations = []
        failures = 0

        for phrase in selected_phrases:
            res: SynthesisResult = await loop.run_in_executor(
                None,
                lambda e=engine, p=phrase: e.synthesize(text=p, output_dir=OUTPUTS_DIR)
            )
            if res.status == "success":
                latencies.append(res.latency_ms)
                rtfs.append(res.real_time_factor)
                durations.append(res.audio_duration_s)
            else:
                failures += 1

        avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else 0.0
        avg_rtf = round(sum(rtfs) / len(rtfs), 4) if rtfs else 0.0
        total_audio_s = round(sum(durations), 2)

        summary[eid] = {
            "engine_name": engine.name,
            "model_size": engine.model_size,
            "avg_latency_ms": avg_latency,
            "avg_rtf": avg_rtf,
            "total_phrases": len(selected_phrases),
            "successful_phrases": len(latencies),
            "failures": failures,
            "total_audio_duration_s": total_audio_s
        }

    return {"category": category_name, "summary": summary}

if __name__ == "__main__":
    import uvicorn
    print("[WebUI] Starting server on http://127.0.0.1:7860 ...", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="info")

