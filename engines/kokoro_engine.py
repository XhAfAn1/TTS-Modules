import os
import time
import urllib.request
from typing import Optional, List, Dict, Any
from .base import BaseTTSEngine, SynthesisResult

KOKORO_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
KOKORO_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

KOKORO_VOICES = [
    {"id": "af_heart", "name": "Heart (US Female - Warm & Expressive)"},
    {"id": "af_bella", "name": "Bella (US Female - Clear & Conversational)"},
    {"id": "af_nicole", "name": "Nicole (US Female - Soft & Caring)"},
    {"id": "af_sarah", "name": "Sarah (US Female - Natural)"},
    {"id": "am_adam", "name": "Adam (US Male - Grounded & Friendly)"},
    {"id": "am_michael", "name": "Michael (US Male - Professional)"},
    {"id": "bf_emma", "name": "Emma (UK Female - Crisp British)"},
    {"id": "bm_george", "name": "George (UK Male - Warm British)"}
]

class KokoroEngine(BaseTTSEngine):
    def __init__(self, models_dir: str = "models/kokoro"):
        super().__init__(
            engine_id="kokoro",
            name="Kokoro-82M",
            model_size="82M parameters (~330MB ONNX / 82M weights)",
            description="Ultra-lightweight open weights with human-level prosody and natural intonation. Highly suitable for AAC."
        )
        self.models_dir = models_dir
        self.voices = KOKORO_VOICES
        self.current_voice = "af_heart"
        self.kokoro_instance = None
        self.model_path = os.path.join(self.models_dir, "kokoro-v1.0.onnx")
        self.voices_path = os.path.join(self.models_dir, "voices-v1.0.bin")
        os.makedirs(self.models_dir, exist_ok=True)

    def _ensure_downloaded(self):
        if not os.path.exists(self.model_path):
            print("[Kokoro] Downloading kokoro-v1.0.onnx...")
            urllib.request.urlretrieve(KOKORO_MODEL_URL, self.model_path)
        if not os.path.exists(self.voices_path):
            print("[Kokoro] Downloading voices-v1.0.bin...")
            urllib.request.urlretrieve(KOKORO_VOICES_URL, self.voices_path)

    def load_model(self, device: str = "auto") -> bool:
        try:
            from kokoro_onnx import Kokoro
            self._ensure_downloaded()
            self.kokoro_instance = Kokoro(self.model_path, self.voices_path)
            self.is_loaded = True
            return True
        except Exception as e:
            # Fallback to KPipeline if kokoro torch package is used
            try:
                from kokoro import KPipeline
                self.kokoro_instance = KPipeline(lang_code='a')
                self.is_loaded = True
                return True
            except Exception as e2:
                print(f"[Kokoro] Error loading model: {e} / {e2}")
                self.is_loaded = False
                return False

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        reference_audio: Optional[str] = None,
        output_dir: str = "outputs"
    ) -> SynthesisResult:
        if not self.is_loaded:
            if not self.load_model():
                return SynthesisResult(
                    engine_id=self.engine_id,
                    engine_name=self.name,
                    text=text,
                    audio_path="",
                    audio_url="",
                    sample_rate=24000,
                    audio_duration_s=0.0,
                    latency_ms=0.0,
                    real_time_factor=0.0,
                    status="error",
                    error="Kokoro model not initialized"
                )

        import soundfile as sf
        if voice == "male":
            chosen_voice = "am_adam"
        elif voice == "female":
            chosen_voice = "af_heart"
        else:
            chosen_voice = voice if voice else self.current_voice
        os.makedirs(output_dir, exist_ok=True)
        filename = f"kokoro_{int(time.time() * 1000)}.wav"
        output_path = os.path.join(output_dir, filename)

        start_time = time.perf_counter()
        try:
            if hasattr(self.kokoro_instance, "create"):
                # kokoro-onnx
                samples, sample_rate = self.kokoro_instance.create(
                    text,
                    voice=chosen_voice,
                    speed=speed,
                    lang="en-us"
                )
                latency_ms = (time.perf_counter() - start_time) * 1000
                sf.write(output_path, samples, sample_rate)
                duration_s = len(samples) / float(sample_rate)
            else:
                # KPipeline generator
                generator = self.kokoro_instance(text, voice=chosen_voice, speed=speed)
                all_audio = []
                sample_rate = 24000
                for _, _, audio in generator:
                    all_audio.append(audio)
                import numpy as np
                combined = np.concatenate(all_audio) if all_audio else np.array([])
                latency_ms = (time.perf_counter() - start_time) * 1000
                sf.write(output_path, combined, sample_rate)
                duration_s = len(combined) / float(sample_rate)

            rtf = (latency_ms / 1000.0) / duration_s if duration_s > 0 else 0.0

            return SynthesisResult(
                engine_id=self.engine_id,
                engine_name=self.name,
                text=text,
                audio_path=output_path,
                audio_url=f"/outputs/{filename}",
                sample_rate=sample_rate,
                audio_duration_s=round(duration_s, 2),
                latency_ms=round(latency_ms, 1),
                real_time_factor=round(rtf, 4),
                status="success"
            )
        except Exception as e:
            return SynthesisResult(
                engine_id=self.engine_id,
                engine_name=self.name,
                text=text,
                audio_path="",
                audio_url="",
                sample_rate=24000,
                audio_duration_s=0.0,
                latency_ms=0.0,
                real_time_factor=0.0,
                status="error",
                error=str(e)
            )

