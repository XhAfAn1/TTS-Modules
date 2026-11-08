import os
import time
import asyncio
import soundfile as sf
from typing import Optional, List, Dict
from .base import BaseTTSEngine, SynthesisResult

class EdgeTTSEngine(BaseTTSEngine):
    def __init__(self):
        super().__init__(
            engine_id="edge_tts",
            name="Edge-TTS (Azure Neural)",
            model_size="Cloud Neural",
            description="Microsoft Azure Edge neural voices. Studio-grade human cadence and expressive prosody (requires internet)."
        )
        self.default_voice = "en-US-JennyNeural"
        self.voices = [
            {"id": "en-US-JennyNeural", "name": "Jenny (US Conversational Female)"},
            {"id": "en-US-GuyNeural", "name": "Guy (US Conversational Male)"},
            {"id": "en-US-AriaNeural", "name": "Aria (US Expressive Female)"},
            {"id": "en-GB-SoniaNeural", "name": "Sonia (British Expressive Female)"},
            {"id": "en-GB-RyanNeural", "name": "Ryan (British Male)"}
        ]

    def load_model(self, device: str = "auto") -> bool:
        self.is_loaded = True
        return True

    def unload_model(self):
        self.is_loaded = False

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        reference_audio: Optional[str] = None,
        output_dir: str = "outputs"
    ) -> SynthesisResult:
        import edge_tts

        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time() * 1000)
        output_filename = f"edge_{timestamp}.mp3"
        output_path = os.path.join(output_dir, output_filename)

        if voice == "male":
            selected_voice = "en-US-GuyNeural"
        elif voice == "female":
            selected_voice = "en-US-JennyNeural"
        else:
            selected_voice = voice if voice and "Neural" in voice else self.default_voice

        # Format speed e.g. "+0%" or "+10%"
        rate_percent = int((speed - 1.0) * 100)
        rate_str = f"+{rate_percent}%" if rate_percent >= 0 else f"{rate_percent}%"

        t_start = time.time()
        try:
            async def _run():
                communicate = edge_tts.Communicate(text, selected_voice, rate=rate_str)
                await communicate.save(output_path)

            asyncio.run(_run())
            t_end = time.time()
            latency_ms = (t_end - t_start) * 1000.0

            audio_info = sf.info(output_path)
            duration_s = audio_info.duration
            sample_rate = audio_info.samplerate
            rtf = (latency_ms / 1000.0) / max(duration_s, 0.001)

            return SynthesisResult(
                engine_id=self.engine_id,
                engine_name=self.name,
                text=text,
                audio_path=output_path,
                audio_url=f"/outputs/{output_filename}",
                sample_rate=sample_rate,
                audio_duration_s=round(duration_s, 2),
                latency_ms=round(latency_ms, 1),
                real_time_factor=round(rtf, 3),
                status="success",
                extra_metrics={"voice": selected_voice, "rate": rate_str}
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

