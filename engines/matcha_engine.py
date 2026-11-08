import os
import time
import urllib.request
import soundfile as sf
from typing import Optional, List, Dict
from .base import BaseTTSEngine, SynthesisResult

class MatchaTTSEngine(BaseTTSEngine):
    def __init__(self, models_dir: str = "models/matcha"):
        super().__init__(
            engine_id="matcha_tts",
            name="Matcha-TTS",
            model_size="~80M",
            description="Optimal-Transport Conditional Flow Matching (OT-CFM). Natural prosody with 3-step ODE flow solver (~120ms latency)."
        )
        self.models_dir = os.path.abspath(models_dir)
        self.acoustic_path = os.path.join(self.models_dir, "model-steps-3.onnx")
        self.vocoder_path = os.path.join(self.models_dir, "vocos-22khz-univ.onnx")
        self.tokens_path = os.path.join(self.models_dir, "tokens.txt")
        self.lexicon_path = os.path.join(self.models_dir, "lexicon-en-us.txt")
        self.tts = None
        self.voices = [
            {"id": "ljspeech", "name": "LJSpeech (Expressive English Female)"}
        ]

    def _ensure_downloaded(self):
        os.makedirs(self.models_dir, exist_ok=True)
        # Check acoustic and tokens
        if not os.path.exists(self.acoustic_path) or not os.path.exists(self.tokens_path):
            tar_url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-en_US-ljspeech.tar.bz2"
            tar_temp = os.path.join(self.models_dir, "matcha_temp.tar.bz2")
            print(f"[Matcha-TTS] Downloading acoustic model from {tar_url}...")
            urllib.request.urlretrieve(tar_url, tar_temp)
            import tarfile
            with tarfile.open(tar_temp, "r:bz2") as tar:
                for member in tar.getmembers():
                    if member.name.endswith("model-steps-3.onnx"):
                        member.name = os.path.basename(member.name)
                        tar.extract(member, path=self.models_dir)
                    elif member.name.endswith("tokens.txt"):
                        member.name = os.path.basename(member.name)
                        tar.extract(member, path=self.models_dir)
            if os.path.exists(tar_temp):
                try:
                    os.remove(tar_temp)
                except Exception:
                    pass

        # Check vocoder
        if not os.path.exists(self.vocoder_path):
            vocoder_url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx"
            print(f"[Matcha-TTS] Downloading Vocos vocoder from {vocoder_url}...")
            urllib.request.urlretrieve(vocoder_url, self.vocoder_path)

        # Check lexicon
        if not os.path.exists(self.lexicon_path):
            lexicon_url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/lexicon-en-us.txt"
            print(f"[Matcha-TTS] Downloading lexicon from {lexicon_url}...")
            urllib.request.urlretrieve(lexicon_url, self.lexicon_path)

    def load_model(self, device: str = "auto") -> bool:
        if self.is_loaded and self.tts is not None:
            return True

        self._ensure_downloaded()

        import sherpa_onnx
        config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                    acoustic_model=self.acoustic_path,
                    vocoder=self.vocoder_path,
                    tokens=self.tokens_path,
                    lexicon=self.lexicon_path
                ),
                num_threads=4,
                provider="cpu"
            )
        )
        self.tts = sherpa_onnx.OfflineTts(config)
        self.is_loaded = True
        return True

    def unload_model(self):
        self.tts = None
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
        output_filename = f"matcha_{timestamp}.wav"
        output_path = os.path.join(output_dir, output_filename)

        t_start = time.time()
        try:
            audio = self.tts.generate(text=text, sid=0, speed=speed)
            t_end = time.time()
            latency_ms = (t_end - t_start) * 1000.0

            if len(audio.samples) == 0:
                raise ValueError("Matcha-TTS generated zero samples.")

            duration_s = len(audio.samples) / audio.sample_rate
            rtf = (latency_ms / 1000.0) / max(duration_s, 0.001)

            sf.write(output_path, audio.samples, audio.sample_rate)

            return SynthesisResult(
                engine_id=self.engine_id,
                engine_name=self.name,
                text=text,
                audio_path=output_path,
                audio_url=f"/outputs/{output_filename}",
                sample_rate=audio.sample_rate,
                audio_duration_s=round(duration_s, 2),
                latency_ms=round(latency_ms, 1),
                real_time_factor=round(rtf, 3),
                status="success",
                extra_metrics={"steps": 3, "solver": "Euler-OT-CFM"}
            )
        except Exception as e:
            return SynthesisResult(
                engine_id=self.engine_id,
                engine_name=self.name,
                text=text,
                audio_path="",
                audio_url="",
                sample_rate=22050,
                audio_duration_s=0.0,
                latency_ms=0.0,
                real_time_factor=0.0,
                status="error",
                error=str(e)
            )

