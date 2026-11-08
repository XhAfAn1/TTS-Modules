import time
import os
import gc
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Generator

@dataclass
class SynthesisResult:
    engine_id: str
    engine_name: str
    text: str
    audio_path: str
    audio_url: str
    sample_rate: int
    audio_duration_s: float
    latency_ms: float
    real_time_factor: float  # generation_time / audio_duration
    memory_used_mb: float = 0.0
    status: str = "success"
    error: Optional[str] = None
    extra_metrics: Dict[str, Any] = field(default_factory=dict)

class BaseTTSEngine:
    def __init__(self, engine_id: str, name: str, model_size: str, description: str):
        self.engine_id = engine_id
        self.name = name
        self.model_size = model_size
        self.description = description
        self.is_loaded = False
        self.device = "cpu"
        self.voices = []

    def load_model(self, device: str = "auto") -> bool:
        """Loads the weights into RAM or VRAM."""
        raise NotImplementedError

    def unload_model(self):
        """Unloads the weights to free memory."""
        self.is_loaded = False
        gc.collect()

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        reference_audio: Optional[str] = None,
        output_dir: str = "outputs"
    ) -> SynthesisResult:
        """Synthesizes text into audio and benchmarks performance."""
        raise NotImplementedError

    def get_available_voices(self) -> List[Dict[str, str]]:
        """Returns a list of supported voices/styles."""
        return self.voices

    def supports_voice_cloning(self) -> bool:
        """Whether this engine supports zero-shot voice cloning via reference audio."""
        return False

    def supports_paralinguistics(self) -> bool:
        """Whether this engine supports tags like [laugh], [sigh], [whisper]."""
        return False

    def supports_streaming(self) -> bool:
        """Whether this engine supports chunked streaming audio output."""
        return False

