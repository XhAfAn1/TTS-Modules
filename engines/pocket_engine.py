import os
import time
from typing import Optional, List, Dict, Any
from .base import BaseTTSEngine, SynthesisResult

POCKET_PRESET_VOICES = [
    {"id": "alba", "name": "Alba (Female Expressive - CALM)"},
    {"id": "anna", "name": "Anna (Female Gentle)"},
    {"id": "cosette", "name": "Cosette (Female Melodic)"},
    {"id": "marius", "name": "Marius (Male Conversational - CALM)"},
    {"id": "jean", "name": "Jean (Male Natural)"},
    {"id": "george", "name": "George (Male Deep)"},
    {"id": "michael", "name": "Michael (Male Clean)"}
]

POCKET_CATALOG_VOICES = {
    # Male
    "marius", "jean", "charles", "paul", "george", "michael",
    "bill_boerst", "peter_yearsley", "stuart_bell", "giovanni",
    "juergen", "rafael", "daan",
    # Female
    "alba", "anna", "cosette", "fantine", "eponine", "azelma",
    "vera", "mary", "jane", "eve", "lola", "estelle", "caro_davy"
}

class PocketTTSEngine(BaseTTSEngine):
    def __init__(self, models_dir: str = "models/pocket"):
        super().__init__(
            engine_id="pocket_tts",
            name="Pocket TTS (Kyutai)",
            model_size="100M parameters",
            description="Continuous Audio Language Model (CALM) by Kyutai. Designed for CPU streaming and instant zero-shot voice cloning."
        )
        self.models_dir = models_dir
        self.voices = POCKET_PRESET_VOICES
        self.current_voice = "alba"
        self.tts_model = None
        self._voice_cache = {}
        os.makedirs(self.models_dir, exist_ok=True)

    def load_model(self, device: str = "auto") -> bool:
        try:
            from pocket_tts import TTSModel
            self.tts_model = TTSModel.load_model()
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"[Pocket TTS] Model load error: {e}")
            self.is_loaded = False
            return False

    def supports_voice_cloning(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

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
                    error="Pocket TTS model not loaded"
                )

        import scipy.io.wavfile
        import numpy as np
        os.makedirs(output_dir, exist_ok=True)
        filename = f"pocket_{int(time.time() * 1000)}.wav"
        output_path = os.path.join(output_dir, filename)

        start_time = time.perf_counter()
        try:
            # Resolve voice target
            has_cloning = getattr(self.tts_model, "has_voice_cloning", False)

            if voice == "male":
                catalog_target = "marius"
            elif voice == "female":
                catalog_target = "alba"
            elif voice and voice in POCKET_CATALOG_VOICES:
                catalog_target = voice
            else:
                catalog_target = self.current_voice

            # If reference audio is provided AND model has voice cloning capability enabled:
            if reference_audio and os.path.exists(reference_audio) and has_cloning:
                prompt_key = reference_audio
            else:
                prompt_key = catalog_target

            # Retrieve or compute voice embedding state
            if prompt_key not in self._voice_cache:
                try:
                    voice_state = self.tts_model.get_state_for_audio_prompt(prompt_key)
                except Exception as ex:
                    # Gated model fallback: use built-in catalog voice
                    print(f"[Pocket TTS] Custom cloning prompt unavailable ({ex}); using catalog voice '{catalog_target}'")
                    prompt_key = catalog_target
                    voice_state = self.tts_model.get_state_for_audio_prompt(prompt_key)
                self._voice_cache[prompt_key] = voice_state
            else:
                voice_state = self._voice_cache[prompt_key]

            # Generate audio tensor
            audio_tensor = self.tts_model.generate_audio(voice_state, text)
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Convert to numpy and save
            audio_data = audio_tensor.cpu().numpy()
            sample_rate = getattr(self.tts_model, "sample_rate", 24000)
            scipy.io.wavfile.write(output_path, sample_rate, audio_data)

            duration_s = len(audio_data) / float(sample_rate)
            rtf = (latency_ms / 1000.0) / duration_s if duration_s > 0 else 0.0

            extra_info = {
                "voice": prompt_key,
                "mode": "Zero-Shot Audio Clone" if (prompt_key == reference_audio) else f"Catalog Voice ({prompt_key})"
            }

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
                status="success",
                extra_metrics=extra_info
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
