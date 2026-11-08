import os
import wave
import struct
import math

def generate_sample_wav(filename="data/samples/default_ref.wav", duration_s=4.0, sample_rate=24000):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    n_samples = int(duration_s * sample_rate)
    
    # Formant-like harmonic frequencies resembling human speech vowel /a/ (fundamental ~130Hz)
    f0 = 130.0
    formants = [(f0, 0.4), (f0 * 2, 0.25), (730.0, 0.2), (1090.0, 0.1), (2440.0, 0.05)]
    
    with wave.open(filename, "w") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        
        frames = bytearray()
        for i in range(n_samples):
            t = float(i) / sample_rate
            # Add subtle natural envelope (rise and fall)
            envelope = math.sin(math.pi * (i / n_samples))
            
            sample_val = 0.0
            for freq, amp in formants:
                sample_val += amp * math.sin(2.0 * math.pi * freq * t)
            
            sample_val *= envelope * 0.8
            # Clamp to 16-bit int range
            int_val = int(max(-1.0, min(1.0, sample_val)) * 32767)
            frames.extend(struct.pack("<h", int_val))
            
        wav_file.writeframes(frames)
    print(f"Generated sample audio: {filename} ({duration_s}s, {sample_rate}Hz mono)")

if __name__ == "__main__":
    generate_sample_wav()

