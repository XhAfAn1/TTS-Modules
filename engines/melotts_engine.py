import os
import time
import soundfile as sf
from typing import Optional, List, Dict
from .base import BaseTTSEngine, SynthesisResult

class MeloTTSEngine(BaseTTSEngine):
    def __init__(self):
        super().__init__(
            engine_id="melo_tts",
            name="MeloTTS (MyShell)",
            model_size="~150M",
            description="Enhanced VITS with prosody modeling. Fast melodic synthesis with multiple regional accents."
        )
        self.model = None
        self.speaker_ids = {}
        self.default_speaker = "EN-US"
        self.voices = [
            {"id": "EN-US", "name": "American English (EN-US)"},
            {"id": "EN-BR", "name": "British English (EN-BR)"},
            {"id": "EN-AU", "name": "Australian English (EN-AU)"},
            {"id": "EN_INDIA", "name": "Indian English (EN_INDIA)"},
            {"id": "EN-Default", "name": "Default English"}
        ]

    def load_model(self, device: str = "auto") -> bool:
        if self.is_loaded and self.model is not None:
            return True

        import torch
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[MeloTTS] Loading MeloTTS on {self.device}...")
        from melo.api import TTS
        self.model = TTS(language="EN", device=self.device)
        raw_spk = self.model.hps.data.spk2id
        if isinstance(raw_spk, dict):
            self.speaker_ids = raw_spk
        else:
            self.speaker_ids = {k: getattr(raw_spk, k) for k in dir(raw_spk) if not k.startswith("_")}
        self.is_loaded = True
        print(f"[MeloTTS] Loaded successfully with speakers: {list(self.speaker_ids.keys())}")
        return True

    def unload_model(self):
        self.model = None
        self.is_loaded = False
        super().unload_model()

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        reference_audio: Optional[str] = None,
        output_dir: str = "outputs"
    ) -> SynthesisResult:
        if not self.is_loaded:
            self.load_model()

        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time() * 1000)
        output_filename = f"melo_{timestamp}.wav"
        output_path = os.path.join(output_dir, output_filename)

        if isinstance(self.speaker_ids, dict):
            speaker_key = voice if voice in self.speaker_ids else self.default_speaker
            speaker_id = self.speaker_ids.get(speaker_key, 0)
        else:
            speaker_key = voice if hasattr(self.speaker_ids, str(voice)) else self.default_speaker
            speaker_id = getattr(self.speaker_ids, speaker_key, 0)

        t_start = time.time()
        try:
            self.model.tts_to_file(text, speaker_id, output_path, speed=speed)
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
                extra_metrics={"speaker": speaker_key, "device": self.device}
            )
        except Exception as e:
            return SynthesisResult(
                engine_id=self.engine_id,
                engine_name=self.name,
                text=text,
                audio_path="",
                audio_url="",
                sample_rate=44100,
                audio_duration_s=0.0,
                latency_ms=0.0,
                real_time_factor=0.0,
                status="error",
                error=str(e)
            )
