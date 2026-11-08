# Real-Time TTS Model Benchmark & Comparison

In this repo, we compare and benchmark popular open-source, lightweight Text-to-Speech (TTS) models that can run locally for real-time speech synthesis.

The testbench includes an interactive WebUI for side-by-side listening, latency measurement, voice profile selection, and zero-shot voice cloning.

---

## Supported Models

| Model | Size | Architecture | Measured Warm Latency | Voice Cloning / Personas | Best For |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Matcha-TTS** | 80M | 3-step Flow Matching ODE | **~32 ms – 70 ms** | LJSpeech | Instant natural prosody with zero diffusion delay |
| **Piper TTS** | ~60MB | VITS / ONNX | **~26 ms – 55 ms** | Multiple voices (Ryan, Lessac) | Ultra-fast offline CPU/embedded synthesis |
| **Piper (High-Fidelity)** | 115MB | High-res VITS ONNX | **~79 ms – 200 ms** | High-res pre-trained voices | Clear articulation with very low latency |
| **MeloTTS** | 150M | Enhanced VITS + Pitch Predictor | **~160 ms – 400 ms** | Regional accents (US/UK/AU/IN) | Melodic conversational flow across accents |
| **Kokoro-82M** | 82M | StyleTTS 2 | **~200 ms – 350 ms** | 8 built-in personas (US/UK) | High naturalness conversational speech |
| **Google TTS** | Cloud | Wavenet / Journey / Neural2 / gTTS | **~430 ms – 860 ms** | Free tier + gTTS fallback | High quality cloud voice with free quota |
| **Edge-TTS** | Cloud | Azure Neural | **~800 ms – 1.0 s** | Microsoft Neural voices | High-definition natural cadence (requires internet) |
| **TikTok TTS** | Cloud | ByteDance Conversational | **~2.2 s – 3.0 s** | 8 casual conversational voices | Free social conversational voices (0 API keys) |
| **Pocket TTS** | 100M | Continuous Audio LM (CALM) | **~1.2 s – 1.4 s** | 25+ built-in catalog voices | Fast CPU streaming with character voice library |
| **Chatterbox Nano** | 110M | 1-step Decoder | **~1.4 s – 1.6 s** | Yes (Zero-shot) | Expressive CPU synthesis with `[laugh]` / `[sigh]` tags |
| **Chatterbox Turbo** | 350M | 1-step Latent Diffusion | **~1.8 s – 2.2 s** | Yes (Zero-shot) | High-fidelity zero-shot voice cloning |
| **Chatterbox Multilingual** | 500M | Multilingual Decoder | **~2.0 s – 2.4 s** | Yes (23+ languages) | Multilingual zero-shot voice cloning |
| **CosyVoice** | 300M | Multi-scale Flow Matching | **~2.5 s – 3.0 s** | Yes (Zero-shot) | High clarity flow matching with streaming |
| **VibeVoice Realtime** | 0.5B | 7.5 Hz Continuous Tokenizer | **~2.5 s – 3.2 s** | Yes (Zero-shot) | Conversational streaming dialogue |

---

## Quick Start

### 1. Requirements
- Python 3.10 or 3.11
- NVIDIA GPU recommended for generative/diffusion models (CUDA 12.x), though Piper, Matcha, and Kokoro run fast on CPU.

### 2. Setup
```bash
# Create and activate virtual environment
python -m venv .venv

# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch WebUI
Double-click `start_webui.bat` or run:
```bash
python webui/server.py
```
Open **`http://127.0.0.1:7860`** in your browser.

---
