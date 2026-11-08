import os
import time
import torch
from typing import Optional, List, Dict, Any
from .base import BaseTTSEngine, SynthesisResult

class ChatterboxNanoEngine(BaseTTSEngine):
    def __init__(self):
        super().__init__(
            engine_id="chatterbox_nano",
            name="Chatterbox Nano",
            model_size="110M parameters",
            description="Ultra-lightweight edge model optimized for CPU inference and zero-shot voice cloning with paralinguistic tag support."
        )
        self.model = None
        self.sample_rate = 24000
        self.voices = [
            {"id": "default", "name": "Default Studio Voice"},
            {"id": "cloned", "name": "Custom Reference Audio (Voice Clone)"}
        ]

    def load_model(self, device: str = "auto") -> bool:
        try:
            from chatterbox.tts import ChatterboxTTS
            dev = "cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device)
            self.device = dev
            print(f"[Chatterbox Nano] Loading on {self.device}...")
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
            self.sample_rate = getattr(self.model, "sr", 24000)
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"[Chatterbox Nano] Load error: {e}")
            self.is_loaded = False
            return False

    def supports_voice_cloning(self) -> bool:
        return True

    def supports_paralinguistics(self) -> bool:
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
                    sample_rate=self.sample_rate,
                    audio_duration_s=0.0,
                    latency_ms=0.0,
                    real_time_factor=0.0,
                    status="error",
                    error="Chatterbox Nano model not loaded."
                )

        import torchaudio as ta
        os.makedirs(output_dir, exist_ok=True)
        filename = f"chatterbox_nano_{int(time.time() * 1000)}.wav"
        output_path = os.path.join(output_dir, filename)

        start_time = time.perf_counter()
        try:
            kwargs = {}
            if reference_audio and os.path.exists(reference_audio):
                kwargs["audio_prompt_path"] = reference_audio

            wav = self.model.generate(text, **kwargs)
            latency_ms = (time.perf_counter() - start_time) * 1000

            if isinstance(wav, torch.Tensor):
                if wav.ndim == 1:
                    wav = wav.unsqueeze(0)
                ta.save(output_path, wav.cpu(), self.sample_rate)
                num_frames = wav.shape[-1]
                duration_s = num_frames / float(self.sample_rate)
            else:
                duration_s = 0.0

            rtf = (latency_ms / 1000.0) / duration_s if duration_s > 0 else 0.0

            return SynthesisResult(
                engine_id=self.engine_id,
                engine_name=self.name,
                text=text,
                audio_path=output_path,
                audio_url=f"/outputs/{filename}",
                sample_rate=self.sample_rate,
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
                sample_rate=self.sample_rate,
                audio_duration_s=0.0,
                latency_ms=0.0,
                real_time_factor=0.0,
                status="error",
                error=str(e)
            )


class ChatterboxTurboEngine(BaseTTSEngine):
    def __init__(self):
        super().__init__(
            engine_id="chatterbox_turbo",
            name="Chatterbox Turbo",
            model_size="350M parameters",
            description="High-speed distilled single-step mel decoder achieving ~200ms TTFA, expressive prosody, and paralinguistic emotional tags."
        )
        self.model = None
        self.sample_rate = 24000
        self.voices = [
            {"id": "default", "name": "Default Studio Voice"},
            {"id": "cloned", "name": "Custom Reference Audio (Voice Clone)"}
        ]

    def load_model(self, device: str = "auto") -> bool:
        try:
            from chatterbox.tts_turbo import ChatterboxTurboTTS, REPO_ID
            from safetensors.torch import load_file
            from huggingface_hub import snapshot_download

            dev = "cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device)
            self.device = dev
            print(f"[Chatterbox Turbo] Loading on {self.device}...")
            self.model = ChatterboxTurboTTS.from_pretrained(device=self.device)

            # Fix upstream ResembleAI checkpoint bug: s3gen_meanflow.safetensors has zeroed mel2wav weights
            # whereas s3gen.safetensors contains the complete trained weights.
            try:
                ckpt_path = snapshot_download(REPO_ID)
                s3gen_path = os.path.join(ckpt_path, "s3gen.safetensors")
                if os.path.exists(s3gen_path):
                    w_gen = load_file(s3gen_path)
                    mel2wav_state = {k.replace("mel2wav.", ""): v for k, v in w_gen.items() if k.startswith("mel2wav.")}
                    self.model.s3gen.mel2wav.load_state_dict(mel2wav_state)
                    self.model.s3gen.to(self.device)
                    print("[Chatterbox Turbo] Successfully patched mel2wav generator weights.")
            except Exception as w_err:
                print(f"[Chatterbox Turbo] Warning during mel2wav weight patching: {w_err}")

            self.sample_rate = getattr(self.model, "sr", 24000)
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"[Chatterbox Turbo] Load error: {e}")
            self.is_loaded = False
            return False

    def supports_voice_cloning(self) -> bool:
        return True

    def supports_paralinguistics(self) -> bool:
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
                    sample_rate=self.sample_rate,
                    audio_duration_s=0.0,
                    latency_ms=0.0,
                    real_time_factor=0.0,
                    status="error",
                    error="Chatterbox Turbo model not loaded."
                )

        import torchaudio as ta
        os.makedirs(output_dir, exist_ok=True)
        filename = f"chatterbox_turbo_{int(time.time() * 1000)}.wav"
        output_path = os.path.join(output_dir, filename)

        start_time = time.perf_counter()
        try:
            kwargs = {}
            if reference_audio and os.path.exists(reference_audio):
                kwargs["audio_prompt_path"] = reference_audio

            wav = self.model.generate(text, **kwargs)
            latency_ms = (time.perf_counter() - start_time) * 1000

            if isinstance(wav, torch.Tensor):
                if wav.ndim == 1:
                    wav = wav.unsqueeze(0)
                ta.save(output_path, wav.cpu(), self.sample_rate)
                num_frames = wav.shape[-1]
                duration_s = num_frames / float(self.sample_rate)
            else:
                duration_s = 0.0

            rtf = (latency_ms / 1000.0) / duration_s if duration_s > 0 else 0.0

            return SynthesisResult(
                engine_id=self.engine_id,
                engine_name=self.name,
                text=text,
                audio_path=output_path,
                audio_url=f"/outputs/{filename}",
                sample_rate=self.sample_rate,
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
                sample_rate=self.sample_rate,
                audio_duration_s=0.0,
                latency_ms=0.0,
                real_time_factor=0.0,
                status="error",
                error=str(e)
            )


class ChatterboxMultilingualEngine(BaseTTSEngine):
    def __init__(self):
        super().__init__(
            engine_id="chatterbox_multilingual",
            name="Chatterbox Multilingual",
            model_size="500M parameters",
            description="Full-scale multilingual model supporting 23+ languages with zero-shot voice cloning across language barriers."
        )
        self.model = None
        self.sample_rate = 24000
        self.voices = [
            {"id": "en", "name": "English"},
            {"id": "es", "name": "Spanish"},
            {"id": "fr", "name": "French"},
            {"id": "de", "name": "German"},
            {"id": "hi", "name": "Hindi"},
            {"id": "ar", "name": "Arabic"}
        ]

    def load_model(self, device: str = "auto") -> bool:
        try:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            dev = "cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device)
            self.device = dev
            print(f"[Chatterbox Multilingual] Loading on {self.device}...")
            self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
            self.sample_rate = getattr(self.model, "sr", 24000)
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"[Chatterbox Multilingual] Load error: {e}")
            self.is_loaded = False
            return False

    def supports_voice_cloning(self) -> bool:
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
                    sample_rate=self.sample_rate,
                    audio_duration_s=0.0,
                    latency_ms=0.0,
                    real_time_factor=0.0,
                    status="error",
                    error="Chatterbox Multilingual model not loaded."
                )

        import torchaudio as ta
        os.makedirs(output_dir, exist_ok=True)
        filename = f"chatterbox_mtl_{int(time.time() * 1000)}.wav"
        output_path = os.path.join(output_dir, filename)

        start_time = time.perf_counter()
        try:
            lang_id = voice if (voice and voice in ["en", "es", "fr", "de", "hi", "ar"]) else "en"
            kwargs = {"language_id": lang_id}
            if reference_audio and os.path.exists(reference_audio):
                kwargs["audio_prompt_path"] = reference_audio

            wav = self.model.generate(text, **kwargs)
            latency_ms = (time.perf_counter() - start_time) * 1000

            if isinstance(wav, torch.Tensor):
                if wav.ndim == 1:
                    wav = wav.unsqueeze(0)
                ta.save(output_path, wav.cpu(), self.sample_rate)
                num_frames = wav.shape[-1]
                duration_s = num_frames / float(self.sample_rate)
            else:
                duration_s = 0.0

            rtf = (latency_ms / 1000.0) / duration_s if duration_s > 0 else 0.0

            return SynthesisResult(
                engine_id=self.engine_id,
                engine_name=self.name,
                text=text,
                audio_path=output_path,
                audio_url=f"/outputs/{filename}",
                sample_rate=self.sample_rate,
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
                sample_rate=self.sample_rate,
                audio_duration_s=0.0,
                latency_ms=0.0,
                real_time_factor=0.0,
                status="error",
                error=str(e)
            )
