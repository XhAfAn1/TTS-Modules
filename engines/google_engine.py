import os
import time
import soundfile as sf
from typing import Optional, List, Dict, Any
from .base import BaseTTSEngine, SynthesisResult

class GoogleTTSEngine(BaseTTSEngine):
    def __init__(self):
        super().__init__(
            engine_id="google_tts",
            name="Google TTS",
            model_size="Cloud Neural",
            description="Google Cloud Text-to-Speech (Journey & Neural2 voices with 1M chars/mo free tier, plus unlimited gTTS fallback)."
        )
        self.voices = [
            {"id": "en-US-Journey-F", "name": "Google Journey Female (Ultra-Natural)"},
            {"id": "en-US-Journey-D", "name": "Google Journey Male (Deep)"},
            {"id": "en-US-Neural2-F", "name": "Google Neural2 Female"},
            {"id": "en-US-Neural2-D", "name": "Google Neural2 Male"},
            {"id": "gtts_en_us", "name": "Google Free Web (Unlimited)"}
        ]
        self.default_voice = "gtts_en_us"
        self.has_cloud_credentials = bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get("GOOGLE_API_KEY"))

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
        output_filename = f"google_{timestamp}.mp3"
        output_path = os.path.join(output_dir, output_filename)

        tld = "com"
        if voice == "male":
            selected_voice = "en-US-Neural2-D" if self.has_cloud_credentials else "gtts_en_us"
            tld = "co.uk"
        elif voice == "female":
            selected_voice = "en-US-Neural2-F" if self.has_cloud_credentials else "gtts_en_us"
            tld = "com"
        else:
            selected_voice = voice if voice else self.default_voice

        t_start = time.perf_counter()

        # Try Google Cloud Neural API if credentials available
        if self.has_cloud_credentials and "Journey" in selected_voice or "Neural2" in selected_voice:
            try:
                from google.cloud import texttospeech
                client = texttospeech.TextToSpeechClient()
                synthesis_input = texttospeech.SynthesisInput(text=text)
                voice_params = texttospeech.VoiceSelectionParams(
                    language_code="en-US",
                    name=selected_voice
                )
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3,
                    speaking_rate=speed
                )
                response = client.synthesize_speech(
                    input=synthesis_input,
                    voice=voice_params,
                    audio_config=audio_config
                )
                with open(output_path, "wb") as out:
                    out.write(response.audio_content)

                latency_ms = (time.perf_counter() - t_start) * 1000
                data, sr = sf.read(output_path)
                duration_s = len(data) / float(sr)
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
                    extra_metrics={"mode": "Google Cloud Neural", "voice": selected_voice}
                )
            except Exception as e:
                # Fallback to gTTS if cloud credentials fail
                print(f"[Google TTS] Cloud API failed ({e}), falling back to free gTTS...")

        # Free Unlimited gTTS Fallback
        try:
            from gtts import gTTS
            tts = gTTS(text=text, lang="en", tld=tld, slow=False)
            tts.save(output_path)

            latency_ms = (time.perf_counter() - t_start) * 1000
            data, sr = sf.read(output_path)
            duration_s = len(data) / float(sr)
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
                extra_metrics={"mode": "Free Unlimited Web", "voice": "en-US"}
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

