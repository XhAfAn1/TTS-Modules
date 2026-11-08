import os
import time
import wave
import urllib.request
from typing import Optional, List, Dict, Any
from .base import BaseTTSEngine, SynthesisResult

PIPER_VOICES = {
    "en_US-lessac-medium": {
        "name": "Lessac Medium (Female, Fast)",
        "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
    },
    "en_US-lessac-high": {
        "name": "Lessac High (Female, High-Fidelity)",
        "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/high/en_US-lessac-high.onnx",
        "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/high/en_US-lessac-high.onnx.json"
    },
    "en_US-ryan-medium": {
        "name": "Ryan (Male, Conversational, Medium)",
        "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/medium/en_US-ryan-medium.onnx",
        "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json"
    },
    "en_US-ryan-high": {
        "name": "Ryan (Male, High-Fidelity)",
        "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high/en_US-ryan-high.onnx",
        "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high/en_US-ryan-high.onnx.json"
    }
}

class PiperEngine(BaseTTSEngine):
    def __init__(self, models_dir: str = "models/piper"):
        super().__init__(
            engine_id="piper",
            name="Piper TTS",
            model_size="~60MB (Voice-dependent)",
            description="Ultra-fast local neural TTS using VITS and ONNX Runtime. Extremely low latency for assistive AAC."
        )
        self.models_dir = models_dir
        self.current_voice_id = "en_US-lessac-medium"
        self.voice_instance = None
        os.makedirs(self.models_dir, exist_ok=True)
        self.voices = [
            {"id": vid, "name": meta["name"]} for vid, meta in PIPER_VOICES.items()
        ]

    def _ensure_model_downloaded(self, voice_id: str):
        if voice_id not in PIPER_VOICES:
            voice_id = "en_US-lessac-medium"
        meta = PIPER_VOICES[voice_id]
        model_path = os.path.join(self.models_dir, f"{voice_id}.onnx")
        config_path = os.path.join(self.models_dir, f"{voice_id}.onnx.json")

        if not os.path.exists(model_path):
            print(f"[Piper] Downloading voice model {voice_id}...")
            urllib.request.urlretrieve(meta["model_url"], model_path)
        if not os.path.exists(config_path):
            print(f"[Piper] Downloading voice config {voice_id}...")
            urllib.request.urlretrieve(meta["config_url"], config_path)
        return model_path, config_path

    def load_model(self, voice_id: str = "en_US-lessac-medium", device: str = "auto") -> bool:
        try:
            from piper import PiperVoice
            model_path, config_path = self._ensure_model_downloaded(voice_id)
            # Piper supports CUDA via onnxruntime execution providers if enabled
            use_cuda = (device == "cuda" or (device == "auto"))
            self.voice_instance = PiperVoice.load(model_path, config_path=config_path, use_cuda=use_cuda)
            self.current_voice_id = voice_id
            self.is_loaded = True
            return True
        except Exception as e:
            # Fallback to CPU if CUDA provider isn't available
            try:
                from piper import PiperVoice
                model_path, config_path = self._ensure_model_downloaded(voice_id)
                self.voice_instance = PiperVoice.load(model_path, config_path=config_path, use_cuda=False)
                self.current_voice_id = voice_id
                self.is_loaded = True
                return True
            except Exception as e2:
                print(f"[Piper] Error loading model: {e2}")
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
        if voice == "male":
            voice = "en_US-ryan-high" if self.engine_id == "piper_high" else "en_US-ryan-medium"
        elif voice == "female":
            voice = "en_US-lessac-high" if self.engine_id == "piper_high" else "en_US-lessac-medium"

        if not self.is_loaded or (voice and voice != self.current_voice_id):
            chosen_voice = voice if voice else self.current_voice_id
            if not self.load_model(voice_id=chosen_voice):
                return SynthesisResult(
                    engine_id=self.engine_id,
                    engine_name=self.name,
                    text=text,
                    audio_path="",
                    audio_url="",
                    sample_rate=0,
                    audio_duration_s=0.0,
                    latency_ms=0.0,
                    real_time_factor=0.0,
                    status="error",
                    error="Failed to load Piper model"
                )

        os.makedirs(output_dir, exist_ok=True)
        filename = f"piper_{int(time.time() * 1000)}.wav"
        output_path = os.path.join(output_dir, filename)

        start_time = time.perf_counter()
        try:
            with wave.open(output_path, "wb") as wav_file:
                self.voice_instance.synthesize_wav(text, wav_file)
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Read sample rate and duration
            with wave.open(output_path, "rb") as wf:
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                duration_s = n_frames / float(sample_rate)

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
                sample_rate=0,
                audio_duration_s=0.0,
                latency_ms=0.0,
                real_time_factor=0.0,
                status="error",
                error=str(e)
            )

    def supports_streaming(self) -> bool:
        return True


class PiperHighEngine(PiperEngine):
    def __init__(self, models_dir: str = "models/piper"):
        super().__init__(models_dir=models_dir)
        self.engine_id = "piper_high"
        self.name = "Piper (High-Fidelity)"
        self.model_size = "~115MB"
        self.description = "High-fidelity VITS neural voice with superior audio resolution and smoother prosody."
        self.current_voice_id = "en_US-lessac-high"


