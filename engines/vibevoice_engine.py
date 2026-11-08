import os
import time
import copy
import glob
import torch
from typing import Optional, List, Dict, Any
from .base import BaseTTSEngine, SynthesisResult

class VibeVoiceRealtimeEngine(BaseTTSEngine):
    def __init__(self, model_id: str = "microsoft/VibeVoice-Realtime-0.5B"):
        super().__init__(
            engine_id="vibevoice_realtime",
            name="VibeVoice Realtime",
            model_size="0.5B parameters (~1.1GB weights)",
            description="Microsoft's ultra-low frame-rate (7.5Hz) continuous tokenizer TTS for expressive streaming speech and real-time dialogue."
        )
        self.model_id = model_id
        self.processor = None
        self.model = None
        self.sample_rate = 24000
        self.voices_dir = os.path.abspath("models/vibevoice/repo_clone/demo/voices/streaming_model")
        self.voice_presets: Dict[str, str] = {}
        self.voices = []
        self._scan_voices()

    def _scan_voices(self):
        self.voice_presets = {}
        self.voices = []
        if not os.path.exists(self.voices_dir):
            try:
                repo_dir = os.path.abspath("models/vibevoice/repo_clone")
                if not os.path.exists(repo_dir):
                    os.makedirs(os.path.dirname(repo_dir), exist_ok=True)
                    import subprocess
                    print("[VibeVoice] Fetching voice presets repository...")
                    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/microsoft/VibeVoice.git", repo_dir], check=True)
            except Exception as e:
                print(f"[VibeVoice] Voice presets auto-clone skipped: {e}")

        if os.path.exists(self.voices_dir):
            for pt_file in glob.glob(os.path.join(self.voices_dir, "*.pt")):
                name = os.path.splitext(os.path.basename(pt_file))[0]
                self.voice_presets[name] = pt_file
                display_name = name.replace("en-", "English ").replace("de-", "German ").replace("fr-", "French ").replace("_", " ")
                self.voices.append({"id": name, "name": display_name})
        if not self.voices:
            self.voices = [
                {"id": "en-Carter_man", "name": "English Carter (Male)"},
                {"id": "en-Emma_woman", "name": "English Emma (Female)"}
            ]

    def load_model(self, device: str = "auto") -> bool:
        repo_dir = os.path.abspath("models/vibevoice/repo_clone")
        if os.path.exists(repo_dir) and repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        try:
            from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
            from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference

            if device == "auto":
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                dev = device
            self.device = dev
            dtype = torch.bfloat16 if dev == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

            print(f"[VibeVoice] Loading processor from {self.model_id}...")
            self.processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_id)

            print(f"[VibeVoice] Loading model on {self.device} (dtype={dtype})...")
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                attn_implementation="sdpa"
            ).to(self.device).eval()

            if hasattr(self.model, "set_ddpm_inference_steps"):
                self.model.set_ddpm_inference_steps(5)

            self.is_loaded = True
            return True
        except Exception as e:
            print(f"[VibeVoice] Load failed: {e}")
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
                    sample_rate=self.sample_rate,
                    audio_duration_s=0.0,
                    latency_ms=0.0,
                    real_time_factor=0.0,
                    status="error",
                    error="VibeVoice Realtime model could not be initialized."
                )

        import soundfile as sf
        from transformers.cache_utils import DynamicCache
        from transformers.modeling_outputs import BaseModelOutputWithPast

        os.makedirs(output_dir, exist_ok=True)
        filename = f"vibevoice_{int(time.time() * 1000)}.wav"
        output_path = os.path.join(output_dir, filename)

        start_time = time.perf_counter()
        try:
            # Select voice prompt
            chosen_voice = voice if (voice and voice in self.voice_presets) else "en-Carter_man"
            voice_path = self.voice_presets.get(chosen_voice)
            if not voice_path or not os.path.exists(voice_path):
                # Fallback to first available preset
                if self.voice_presets:
                    voice_path = next(iter(self.voice_presets.values()))
                else:
                    voice_path = os.path.join(self.voices_dir, "en-Carter_man.pt")

            with torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache]):
                voice_prompt = torch.load(voice_path, map_location=self.device, weights_only=True)

            clean_text = text.replace("’", "'").replace('“', '"').replace('”', '"')
            inputs = self.processor.process_input_with_cached_prompt(
                text=clean_text,
                cached_prompt=voice_prompt,
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    cfg_scale=1.5,
                    tokenizer=self.processor.tokenizer,
                    generation_config={"do_sample": False},
                    all_prefilled_outputs=copy.deepcopy(voice_prompt)
                )

            latency_ms = (time.perf_counter() - start_time) * 1000

            if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
                audio_out = outputs.speech_outputs[0].float().cpu().numpy().squeeze()
                sf.write(output_path, audio_out, self.sample_rate)
                duration_s = len(audio_out) / float(self.sample_rate)
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
