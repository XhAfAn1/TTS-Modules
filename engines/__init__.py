from .base import BaseTTSEngine, SynthesisResult
from .piper_engine import PiperEngine, PiperHighEngine
from .kokoro_engine import KokoroEngine
from .pocket_engine import PocketTTSEngine
from .melotts_engine import MeloTTSEngine
from .edgetts_engine import EdgeTTSEngine
from .google_engine import GoogleTTSEngine
from .tiktok_engine import TikTokTTSEngine
from .matcha_engine import MatchaTTSEngine
from .cosyvoice_engine import CosyVoiceEngine
from .chatterbox_engines import (
    ChatterboxNanoEngine,
    ChatterboxTurboEngine,
    ChatterboxMultilingualEngine
)
from .vibevoice_engine import VibeVoiceRealtimeEngine

ENGINE_REGISTRY = {
    "piper": PiperEngine,
    "piper_high": PiperHighEngine,
    "matcha_tts": MatchaTTSEngine,
    "kokoro": KokoroEngine,
    "melo_tts": MeloTTSEngine,
    "cosyvoice": CosyVoiceEngine,
    "edge_tts": EdgeTTSEngine,
    "google_tts": GoogleTTSEngine,
    "tiktok_tts": TikTokTTSEngine,
    "pocket_tts": PocketTTSEngine,
    "chatterbox_nano": ChatterboxNanoEngine,
    "chatterbox_turbo": ChatterboxTurboEngine,
    "chatterbox_multilingual": ChatterboxMultilingualEngine,
    "vibevoice_realtime": VibeVoiceRealtimeEngine
}

ENGINE_METADATA = [
    {
        "id": "piper",
        "name": "Piper TTS (Standard)",
        "size": "Voice-dep (~60MB)",
        "strength": "Instant offline synthesis, ~30ms latency on CPU",
        "aac_suitability": "Maximum (Instant)",
        "supports_voice_cloning": False,
        "supports_paralinguistics": False,
        "category": "Edge / Offline"
    },
    {
        "id": "piper_high",
        "name": "Piper (High-Fidelity)",
        "size": "115MB",
        "strength": "High-fidelity neural voice, smooth prosody, ~79ms latency",
        "aac_suitability": "Maximum (Instant)",
        "supports_voice_cloning": False,
        "supports_paralinguistics": False,
        "category": "Edge / Offline"
    },
    {
        "id": "matcha_tts",
        "name": "Matcha-TTS",
        "size": "80M",
        "strength": "3-step Flow Matching ODE, natural intonation, ~60-120ms latency",
        "aac_suitability": "Maximum (Instant)",
        "supports_voice_cloning": False,
        "supports_paralinguistics": False,
        "category": "Flow Matching Fast"
    },
    {
        "id": "kokoro",
        "name": "Kokoro-82M",
        "size": "82M",
        "strength": "Lightweight, 8 natural voice profiles, human-like cadence (~350ms)",
        "aac_suitability": "Very High",
        "supports_voice_cloning": False,
        "supports_paralinguistics": False,
        "category": "Conversational"
    },
    {
        "id": "melo_tts",
        "name": "MeloTTS",
        "size": "150M",
        "strength": "Melodic pitch variance, multi-accent (US/UK/AU/IN), ~400ms latency",
        "aac_suitability": "Very High",
        "supports_voice_cloning": False,
        "supports_paralinguistics": False,
        "category": "Conversational Fast"
    },
    {
        "id": "edge_tts",
        "name": "Edge-TTS (Azure Neural)",
        "size": "Cloud Neural",
        "strength": "Studio-quality neural voices (Jenny/Guy), 100% free unlimited",
        "aac_suitability": "High (Requires Wi-Fi)",
        "supports_voice_cloning": False,
        "supports_paralinguistics": False,
        "category": "Cloud Neural"
    },
    {
        "id": "google_tts",
        "name": "Google TTS (Neural/Free)",
        "size": "Cloud Neural",
        "strength": "1M chars/mo free tier (Journey/Neural2) + unlimited free gTTS",
        "aac_suitability": "High (Requires Wi-Fi)",
        "supports_voice_cloning": False,
        "supports_paralinguistics": False,
        "category": "Cloud Neural"
    },
    {
        "id": "tiktok_tts",
        "name": "TikTok TTS (Conversational)",
        "size": "Cloud Neural",
        "strength": "100% free conversational modern voices, zero API keys",
        "aac_suitability": "High (Requires Wi-Fi)",
        "supports_voice_cloning": False,
        "supports_paralinguistics": False,
        "category": "Cloud Neural"
    },
    {
        "id": "cosyvoice",
        "name": "CosyVoice",
        "size": "300M",
        "strength": "Multi-scale flow matching, high vocal clarity & streaming",
        "aac_suitability": "High",
        "supports_voice_cloning": True,
        "supports_paralinguistics": True,
        "category": "Flow Matching"
    },
    {
        "id": "pocket_tts",
        "name": "Pocket TTS",
        "size": "100M",
        "strength": "CPU streaming, continuous audio language model, voice cloning",
        "aac_suitability": "Very High",
        "supports_voice_cloning": True,
        "supports_paralinguistics": False,
        "category": "Edge Streaming / Clone"
    },
    {
        "id": "chatterbox_nano",
        "name": "Chatterbox Nano",
        "size": "110M",
        "strength": "CPU inference, single-step decoder, voice cloning & tags",
        "aac_suitability": "High",
        "supports_voice_cloning": True,
        "supports_paralinguistics": True,
        "category": "Edge / Expressive"
    },
    {
        "id": "chatterbox_turbo",
        "name": "Chatterbox Turbo",
        "size": "350M",
        "strength": "Fast single-step decoder, expressive speech, paralinguistic tags",
        "aac_suitability": "High",
        "supports_voice_cloning": True,
        "supports_paralinguistics": True,
        "category": "Expressive Realtime"
    },
    {
        "id": "chatterbox_multilingual",
        "name": "Chatterbox Multilingual",
        "size": "500M",
        "strength": "Multilingual (23+ languages), zero-shot voice cloning",
        "aac_suitability": "Medium",
        "supports_voice_cloning": True,
        "supports_paralinguistics": False,
        "category": "Multilingual"
    },
    {
        "id": "vibevoice_realtime",
        "name": "VibeVoice Realtime",
        "size": "0.5B",
        "strength": "7.5Hz continuous tokenizer, real-time streaming dialogue",
        "aac_suitability": "Medium-High",
        "supports_voice_cloning": True,
        "supports_paralinguistics": True,
        "category": "Frontier Streaming"
    }
]

ENGINE_VOICES = {
    "piper": [
        {"id": "en_US-lessac-medium", "name": "Lessac (Female - Fast)", "gender": "female"},
        {"id": "en_US-ryan-medium", "name": "Ryan (Male - Conversational)", "gender": "male"}
    ],
    "piper_high": [
        {"id": "en_US-lessac-high", "name": "Lessac High (Female - High-Res)", "gender": "female"},
        {"id": "en_US-ryan-high", "name": "Ryan High (Male - High-Res)", "gender": "male"}
    ],
    "matcha_tts": [
        {"id": "ljspeech", "name": "LJSpeech (Female - Flow Matching)", "gender": "female"}
    ],
    "kokoro": [
        {"id": "af_heart", "name": "Heart (US Female - Warm)", "gender": "female"},
        {"id": "af_bella", "name": "Bella (US Female - Clear)", "gender": "female"},
        {"id": "af_nicole", "name": "Nicole (US Female - Soft)", "gender": "female"},
        {"id": "af_sarah", "name": "Sarah (US Female - Natural)", "gender": "female"},
        {"id": "bf_emma", "name": "Emma (UK Female - British)", "gender": "female"},
        {"id": "am_adam", "name": "Adam (US Male - Grounded & Friendly)", "gender": "male"},
        {"id": "am_michael", "name": "Michael (US Male - Professional)", "gender": "male"},
        {"id": "bm_george", "name": "George (UK Male - British)", "gender": "male"}
    ],
    "melo_tts": [
        {"id": "EN-US", "name": "American English (EN-US)", "gender": "female"},
        {"id": "EN-BR", "name": "British English (EN-BR)", "gender": "female"},
        {"id": "EN-AU", "name": "Australian English (EN-AU)", "gender": "female"},
        {"id": "EN_INDIA", "name": "Indian English (EN_INDIA)", "gender": "female"},
        {"id": "EN-Default", "name": "Default English", "gender": "both"}
    ],
    "edge_tts": [
        {"id": "en-US-JennyNeural", "name": "Jenny (US Female - Conversational)", "gender": "female"},
        {"id": "en-US-AriaNeural", "name": "Aria (US Female - Expressive)", "gender": "female"},
        {"id": "en-GB-SoniaNeural", "name": "Sonia (UK Female - British)", "gender": "female"},
        {"id": "en-US-GuyNeural", "name": "Guy (US Male - Conversational)", "gender": "male"},
        {"id": "en-GB-RyanNeural", "name": "Ryan (UK Male - British)", "gender": "male"},
        {"id": "en-US-ChristopherNeural", "name": "Christopher (US Male - Professional)", "gender": "male"}
    ],
    "google_tts": [
        {"id": "gtts_en_us", "name": "Google Web Free (Unlimited Female)", "gender": "female"},
        {"id": "en-US-Journey-F", "name": "Google Journey Female (Ultra-Natural)", "gender": "female"},
        {"id": "en-US-Neural2-F", "name": "Google Neural2 Female", "gender": "female"},
        {"id": "en-US-Journey-D", "name": "Google Journey Male (Deep)", "gender": "male"},
        {"id": "en-US-Neural2-D", "name": "Google Neural2 Male", "gender": "male"}
    ],
    "tiktok_tts": [
        {"id": "en_us_001", "name": "TikTok Female 1 (Classic Viral)", "gender": "female"},
        {"id": "en_au_001", "name": "TikTok AU Female", "gender": "female"},
        {"id": "en_us_006", "name": "TikTok Male 1 (Conversational)", "gender": "male"},
        {"id": "en_us_007", "name": "TikTok Male 2 (Storyteller)", "gender": "male"},
        {"id": "en_us_009", "name": "TikTok Male 3 (Warm Tone)", "gender": "male"},
        {"id": "en_uk_001", "name": "TikTok UK Male 1", "gender": "male"},
        {"id": "en_uk_003", "name": "TikTok UK Male 2", "gender": "male"},
        {"id": "en_au_002", "name": "TikTok AU Male", "gender": "male"}
    ],
    "pocket_tts": [
        {"id": "alba", "name": "Alba (Female - Expressive Flagship)", "gender": "female"},
        {"id": "anna", "name": "Anna (Female - Gentle)", "gender": "female"},
        {"id": "cosette", "name": "Cosette (Female - Melodic)", "gender": "female"},
        {"id": "fantine", "name": "Fantine (Female - Soft)", "gender": "female"},
        {"id": "eponine", "name": "Eponine (Female - Clear)", "gender": "female"},
        {"id": "vera", "name": "Vera (Female - Conversational)", "gender": "female"},
        {"id": "mary", "name": "Mary (Female - Warm)", "gender": "female"},
        {"id": "jane", "name": "Jane (Female - Grounded)", "gender": "female"},
        {"id": "eve", "name": "Eve (Female - Professional)", "gender": "female"},
        {"id": "lola", "name": "Lola (Female - Spanish accent)", "gender": "female"},
        {"id": "estelle", "name": "Estelle (Female - French accent)", "gender": "female"},
        {"id": "marius", "name": "Marius (Male - Conversational Flagship)", "gender": "male"},
        {"id": "javert", "name": "Javert (Male - Deep Resonant)", "gender": "male"},
        {"id": "jean", "name": "Jean (Male - Natural)", "gender": "male"},
        {"id": "charles", "name": "Charles (Male - Articulate)", "gender": "male"},
        {"id": "paul", "name": "Paul (Male - Warm)", "gender": "male"},
        {"id": "george", "name": "George (Male - Deep)", "gender": "male"},
        {"id": "michael", "name": "Michael (Male - Clean)", "gender": "male"},
        {"id": "bill_boerst", "name": "Bill Boerst (Male - Narrative)", "gender": "male"},
        {"id": "peter_yearsley", "name": "Peter Yearsley (Male - British)", "gender": "male"},
        {"id": "stuart_bell", "name": "Stuart Bell (Male - Scottish)", "gender": "male"},
        {"id": "giovanni", "name": "Giovanni (Male - Italian accent)", "gender": "male"},
        {"id": "juergen", "name": "Juergen (Male - German accent)", "gender": "male"},
        {"id": "rafael", "name": "Rafael (Male - Warm accent)", "gender": "male"},
        {"id": "daan", "name": "Daan (Male - Dutch accent)", "gender": "male"}
    ],
    "chatterbox_nano": [
        {"id": "cloned", "name": "Zero-Shot Reference Voice Clone", "gender": "both"}
    ],
    "chatterbox_turbo": [
        {"id": "cloned", "name": "Zero-Shot Reference Voice Clone", "gender": "both"}
    ],
    "chatterbox_multilingual": [
        {"id": "cloned", "name": "Zero-Shot Reference Voice Clone", "gender": "both"}
    ],
    "cosyvoice": [
        {"id": "cloned", "name": "Zero-Shot Flow Matching Clone", "gender": "both"}
    ],
    "vibevoice_realtime": [
        {"id": "cloned", "name": "Zero-Shot Streaming Clone", "gender": "both"}
    ]
}
