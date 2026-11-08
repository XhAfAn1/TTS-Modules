import os
import sys
import time
import gc
from typing import Optional, List, Dict, Any
import torch

from .base import BaseTTSEngine, SynthesisResult

class CosyVoiceEngine(BaseTTSEngine):
    def __init__(self, model_dir: str = "pretrained_models/CosyVoice-300M-SFT"):
        super().__init__(
            engine_id="cosyvoice",
            name="CosyVoice",
            model_size="300M",
            description="Multi-scale flow matching with ultra-natural prosody and streaming generation."
        )
        self.model_dir = model_dir
        self.model_instance = None
        self.voices = [
            {"id": "英文女", "name": "English Female (Natural Warm)"},
            {"id": "英文男", "name": "English Male (Deep Conversational)"},
            {"id": "中文女", "name": "Chinese Female (Warm)"},
            {"id": "中文男", "name": "Chinese Male (Clear)"},
            {"id": "日语男", "name": "Japanese Male"},
            {"id": "粤语女", "name": "Cantonese Female"},
            {"id": "韩语女", "name": "Korean Female"}
        ]
        self.default_voice = "英文女"

    def load_model(self, device: str = "auto") -> bool:
        if self.is_loaded and self.model_instance is not None:
            return True

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cosy_root = os.path.join(base_dir, "models", "cosyvoice")
        matcha_root = os.path.join(cosy_root, "third_party", "Matcha-TTS")

        if not os.path.exists(cosy_root):
            try:
                import subprocess
                print(f"[CosyVoice] Cloning repository into {cosy_root}...")
                subprocess.run(["git", "clone", "--recursive", "https://github.com/FunAudioLLM/CosyVoice.git", cosy_root], check=True)
            except Exception as e:
                print(f"[CosyVoice] Auto-clone skipped or failed: {e}")

        if os.path.exists(cosy_root) and cosy_root not in sys.path:
            sys.path.insert(0, cosy_root)
        if os.path.exists(matcha_root) and matcha_root not in sys.path:
            sys.path.insert(0, matcha_root)

        abs_model_dir = os.path.abspath(os.path.join(base_dir, self.model_dir))
        if not os.path.exists(abs_model_dir):
            try:
                from modelscope import snapshot_download
                abs_model_dir = snapshot_download('iic/CosyVoice-300M-SFT', local_dir=abs_model_dir)
            except Exception as e:
                print(f"[CosyVoice] Error downloading model: {e}")
                return False

        try:
            from cosyvoice.cli.cosyvoice import CosyVoice
            print(f"[CosyVoice] Initializing model from {abs_model_dir}...")
            self.model_instance = CosyVoice(abs_model_dir)
            available_spks = self.model_instance.list_available_spks()
            if available_spks:
                self.voices = [{"id": spk, "name": spk} for spk in available_spks]
                self.default_voice = available_spks[0]
                # If English female or male in spks, prefer as default
                for spk in available_spks:
                    if "英" in spk or "en" in spk.lower():
                        self.default_voice = spk
                        break
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"[CosyVoice] Failed to load model: {e}")
            self.is_loaded = False
            return False

    def unload_model(self):
        if self.model_instance is not None:
            del self.model_instance
            self.model_instance = None
        self.is_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        super().unload_model()

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        reference_audio: Optional[str] = None,
        output_dir: str = "outputs"
    ) -> SynthesisResult:
        if not self.is_loaded or self.model_instance is None:
            if not self.load_model():
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
                    error="CosyVoice model failed to initialize."
                )

        selected_voice = voice if voice and voice in [v["id"] for v in self.voices] else self.default_voice

        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time() * 1000)
        output_filename = f"cosyvoice_{timestamp}.wav"
        output_path = os.path.join(output_dir, output_filename)

        import torchaudio
        start_time = time.perf_counter()
        try:
            chunks = []
            for out in self.model_instance.inference_sft(text, selected_voice, stream=False, speed=speed):
                chunks.append(out['tts_speech'])

            if not chunks:
                raise ValueError("CosyVoice returned no audio output.")

            full_audio = torch.cat(chunks, dim=1)
            torchaudio.save(output_path, full_audio, self.model_instance.sample_rate)
            latency_ms = (time.perf_counter() - start_time) * 1000

            duration_s = full_audio.shape[1] / float(self.model_instance.sample_rate)
            rtf = (latency_ms / 1000.0) / max(duration_s, 0.001)

            return SynthesisResult(
                engine_id=self.engine_id,
                engine_name=self.name,
                text=text,
                audio_path=output_path,
                audio_url=f"/outputs/{output_filename}",
                sample_rate=self.model_instance.sample_rate,
                audio_duration_s=round(duration_s, 2),
                latency_ms=round(latency_ms, 1),
                real_time_factor=round(rtf, 4),
                status="success",
                extra_metrics={"voice": selected_voice, "sample_rate": self.model_instance.sample_rate}
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

    def supports_streaming(self) -> bool:
        return True
