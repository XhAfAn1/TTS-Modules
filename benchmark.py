import os
import sys
import json
import time
from typing import List, Dict, Any

from engines import ENGINE_REGISTRY, ENGINE_METADATA, BaseTTSEngine, SynthesisResult

BENCHMARK_PROMPTS = [
    {
        "category": "Quick Response (Short)",
        "text": "Yes, thank you very much."
    },
    {
        "category": "Urgent Care (Medium)",
        "text": "I need some water, please. Can you help me adjust my position?"
    },
    {
        "category": "Conversational (Long)",
        "text": "Good morning! It is really nice to see you today. How was your day? Tell me everything."
    },
    {
        "category": "Expressive (Paralinguistic)",
        "text": "Well, that was quite a surprise! [laugh] I did not expect that at all."
    }
]

def run_benchmarks(engine_ids: List[str] = None, output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)
    if not engine_ids:
        engine_ids = list(ENGINE_REGISTRY.keys())

    print("=" * 75)
    print(" REAL-TIME TTS BENCHMARK & EVALUATION SUITE")
    print("=" * 75)

    all_results = {}

    for eid in engine_ids:
        if eid not in ENGINE_REGISTRY:
            print(f"[SKIP] Unknown engine id: {eid}")
            continue

        print(f"\n---> Benchmarking Engine: {eid} <---")
        engine_cls = ENGINE_REGISTRY[eid]
        engine: BaseTTSEngine = engine_cls()

        # Measure Cold Start Load Time
        t0 = time.perf_counter()
        loaded = engine.load_model()
        load_time_s = round(time.perf_counter() - t0, 2)

        if not loaded:
            print(f"[ERROR] Failed to load {eid}. Skipping.")
            all_results[eid] = {
                "name": engine.name,
                "loaded": False,
                "load_time_s": load_time_s,
                "trials": []
            }
            continue

        print(f"Loaded {engine.name} in {load_time_s}s.")

        trials = []
        for prompt in BENCHMARK_PROMPTS:
            cat = prompt["category"]
            txt = prompt["text"]
            res: SynthesisResult = engine.synthesize(txt, output_dir=output_dir)

            if res.status == "success":
                print(f"  [{cat}] Latency: {res.latency_ms:.1f}ms | RTF: {res.real_time_factor:.4f}x | Audio: {res.audio_duration_s:.2f}s")
                trials.append({
                    "category": cat,
                    "text": txt,
                    "latency_ms": res.latency_ms,
                    "rtf": res.real_time_factor,
                    "audio_duration_s": res.audio_duration_s,
                    "audio_path": res.audio_path
                })
            else:
                print(f"  [{cat}] Synthesis FAILED: {res.error}")
                trials.append({
                    "category": cat,
                    "text": txt,
                    "status": "error",
                    "error": res.error
                })

        valid_latencies = [t["latency_ms"] for t in trials if "latency_ms" in t]
        valid_rtfs = [t["rtf"] for t in trials if "rtf" in t]

        engine.unload_model()

        all_results[eid] = {
            "name": engine.name,
            "model_size": engine.model_size,
            "load_time_s": load_time_s,
            "avg_latency_ms": round(sum(valid_latencies) / len(valid_latencies), 1) if valid_latencies else 0.0,
            "avg_rtf": round(sum(valid_rtfs) / len(valid_rtfs), 4) if valid_rtfs else 0.0,
            "trials": trials
        }

    # Summary Table
    print("\n" + "=" * 75)
    print(f"{'Engine':<25} | {'Size':<16} | {'Avg Latency':<12} | {'Avg RTF':<10} | {'Status'}")
    print("-" * 75)
    for eid, data in all_results.items():
        if data.get("loaded", True) and data.get("avg_latency_ms", 0) > 0:
            print(f"{data['name']:<25} | {data.get('model_size', 'N/A'):<16} | {data['avg_latency_ms']:>8.1f} ms | {data['avg_rtf']:>8.4f}x | OK")
        else:
            print(f"{data['name']:<25} | {data.get('model_size', 'N/A'):<16} | {'N/A':>8}    | {'N/A':>8}   | FAILED")
    print("=" * 75)

    report_path = os.path.join(output_dir, "benchmark_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull benchmark report saved to: {report_path}")

if __name__ == "__main__":
    target_engines = sys.argv[1:] if len(sys.argv) > 1 else None
    run_benchmarks(target_engines)

