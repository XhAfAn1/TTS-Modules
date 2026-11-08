import os
import time
import base64
import requests
import soundfile as sf
from typing import Optional, List, Dict, Any
from .base import BaseTTSEngine, SynthesisResult

TIKTOK_VOICES = [
    {"id": "en_us_001", "name": "TikTok Female (Classic Viral)"},
    {"id": "en_us_006", "name": "TikTok Male 1 (Conversational)"},
    {"id": "en_us_007", "name": "TikTok Male 2 (Storyteller)"},
    {"id": "en_us_009", "name": "TikTok Male 3 (Warm Tone)"},
    {"id": "en_uk_001", "name": "TikTok UK Male 1"},
    {"id": "en_uk_003", "name": "TikTok UK Male 2"},
    {"id": "en_au_001", "name": "TikTok AU Female"},
    {"id": "en_au_002", "name": "TikTok AU Male"}
]

class TikTokTTSEngine(BaseTTSEngine):
    def __init__(self):
        super().__init__(
            engine_id="tiktok_tts",
            name="TikTok TTS",
            model_size="Cloud Neural",
            description="100% Free natural conversational social voices with zero API key requirement."
        )
        self.voices = TIKTOK_VOICES
        self.default_voice = "en_us_001"

    def load_model(self, device: str = "auto") -> bool:
        self.is_loaded = True
        return True

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        reference_audio: Optional[str] = None,
        output_dir: str = "outputs"
    ) -> SynthesisResult:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time() * 1000)
        output_filename = f"tiktok_{timestamp}.mp3"
        output_path = os.path.join(output_dir, output_filename)

        if voice == "male":
            selected_voice = "en_us_006"
        elif voice == "female":
            selected_voice = "en_us_001"
        else:
            selected_voice = voice if voice else self.default_voice

        t_start = time.perf_counter()

        try:
            url = "https://tiktok-tts.weilnet.workers.dev/api/generation"
            payload = {"text": text, "voice": selected_voice}
            resp = requests.post(url, json=payload, timeout=12)
            
            if resp.status_code != 200:
                raise ValueError(f"HTTP {resp.status_code}: {resp.text[:100]}")
            
            data = resp.json()
            if not data.get("success") or "data" not in data:
                raise ValueError(data.get("error", "Failed to generate TikTok speech"))

            audio_bytes = base64.b64decode(data["data"])
            with open(output_path, "wb") as f:
                f.write(audio_bytes)

            latency_ms = (time.perf_counter() - t_start) * 1000
            audio_data, sr = sf.read(output_path)
            duration_s = len(audio_data) / float(sr)
            rtf = (latency_ms / 1000.0) / max(duration_s, 0.001)

            return SynthesisResult(
                engine_id=self.engine_id,
                engine_name=self.name,
                text=text,
                audio_path=output_path,
                audio_url=f"/outputs/{output_filename}",
                sample_rate=sr,
                audio_duration_s=round(duration_s, 2),
                latency_ms=round(latency_ms, 1),
                real_time_factor=round(rtf, 4),
                status="success",
                extra_metrics={"voice": selected_voice, "cloud": "TikTok Neural"}
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

    def supports_streaming(self) -> bool:
        return False

