import os
from elevenlabs.client import ElevenLabs
from dotenv import dotenv_values
# We only need the 'save' function for saving the audio
from elevenlabs import save 

# --- 1. Configuration ---

config = dotenv_values(".env")

ELEVENLABS_API_KEY = config.get("ELEVENLABS_API_KEY")

# Define Speech Parameters and Paths
TEXT_TO_SPEAK = "Hi! This is Eleven fron Stranger Things."
VOICE_ID = "pNInz6obpgDQGcFmaJgB"  # Example Voice ID
MODEL_ID = "eleven_multilingual_v2" 

# Define the folder and the filename
OUTPUT_FOLDER = "output_audio_files"
BASE_FILENAME = "elevenlabs_output2.mp3"
OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, BASE_FILENAME)


# --- 2. Initialize the ElevenLabs Client ---
# The client is initialized directly with the hardcoded API key
print("Initializing ElevenLabs Client...")
try:
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY) 
except Exception as e:
    print(f"Error initializing ElevenLabs client: {e}")
    exit()


# --- 3. Ensure Directory and Generate Audio ---

# Ensure the output directory exists before saving
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print(f"Ensuring output directory '{OUTPUT_FOLDER}' exists.")

print(f"Generating audio for voice ID: {VOICE_ID}")
try:
    # The .convert() method performs the core text-to-speech task
    audio = client.text_to_speech.convert(
        text=TEXT_TO_SPEAK,
        voice_id=VOICE_ID,
        model_id=MODEL_ID,
    )
    print("Audio generation successful.")

except Exception as e:
    # This will catch authorization failures, network issues, etc.
    print(f"🔴 Error during audio generation. Please check your API key and network connection.")
    print(f"Details: {e}")
    exit()


# --- 4. Save the Generated Audio ---

try:
    # Save the audio to the file inside the specified folder
    save(audio, OUTPUT_PATH)
    print(f"\n✅ SUCCESS: Audio saved to **{OUTPUT_PATH}**")
except Exception as e:
    print(f"Error saving file: {e}")