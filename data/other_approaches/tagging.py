import json
import time

from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Automatically finds and loads the .env file variables into the system environment
load_dotenv() 

# Now the official SDK can natively pick up the variable seamlessly
client = genai.Client()


# Define the allowed tags
ALLOWED_MOODS = [
    "aggressive", "confident", "melancholic", "introspective", "motivational", 
    "hype", "dark", "romantic", "vulnerable", "rebellious", "playful", 
    "emotional", "storytelling"
]

ALLOWED_GENRES = [
    "trap", "boom_bap", "drill", "grime", "conscious_rap", "melodic_rap", 
    "experimental", "underground", "mainstream", "alternative_hiphop"
]

ALLOWED_RHYME_SCHEMES = [
    "monorhyme", "couplet", "alternate_rhyme", "internal_rhyme", 
    "multisyllabic", "dense_internal", "freeform"
]

ALLOWED_CADENCES = [
    "slow", "midtempo", "fast", "switching", "staccato", "rolling", "bouncy"
]

class Annotations(BaseModel):
    mood: str = Field(description=f"A single mood strictly from the allowed list: {ALLOWED_MOODS}")
    genre: str = Field(description=f"A single genre strictly from the allowed list: {ALLOWED_GENRES}")
    rhyme_scheme: str = Field(description=f"A single rhyme scheme strictly from the allowed list: {ALLOWED_RHYME_SCHEMES}")
    cadence: str = Field(description=f"A single cadence strictly from the allowed list: {ALLOWED_CADENCES}")
    year: int = Field(description="The accurate release year of the song, determined based on the name of the song and artist.")

class TaggingResult(BaseModel):
    annotations: Annotations

def tag_song_with_gemma4(title: str, artist: str, lyrics: str, max_retries=5):
    truncated_lyrics = lyrics[:2500]
    
    prompt = (
        f"Analyze this rap song: '{title}' by {artist}. "
        f"You MUST select exactly 1 primary mood from: {ALLOWED_MOODS}. "
        f"You MUST select exactly 1 genre from: {ALLOWED_GENRES}. "
        f"You MUST select exactly 1 rhyme scheme from: {ALLOWED_RHYME_SCHEMES}. "
        f"You MUST select exactly 1 cadence from: {ALLOWED_CADENCES}. "
        f"You MUST provide the accurate release year of the song based on the name of the song and artist. "
        "Do not invent any new tags. Your output must strictly conform to the provided schema."
    )
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemma-4-26b-a4b-it", # Using the ultra-fast MoE variant hosted on AI Studio
                contents=[prompt, f"Lyrics:\n{truncated_lyrics}"],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=TaggingResult,
                    system_instruction=(
                        "You are an expert musicologist with 30 years of experience in rap songs and a meticulous tagger. "
                        "You must evaluate the song and output the annotations strictly from the allowed lists. "
                        "Your output must strictly conform to the provided JSON schema."
                    )
                )
            )
            
            # Parse the guaranteed JSON response after cleaning code fences
            cleaned_text = response.text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            elif cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()

            result_data = json.loads(cleaned_text)
            return result_data
            
        except Exception as e:
            print(f"Error calling Google AI Studio (Attempt {attempt+1}/{max_retries}): {e}")
            if 'response' in locals() and hasattr(response, 'text'):
                print("RAW RESPONSE TEXT:")
                print(repr(response.text))
            if attempt < max_retries - 1:
                sleep_time = (2 ** attempt) * 5  # Exponential backoff: 5, 10, 20, 40 seconds
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("Max retries reached. Returning None.")
                return None

# --- Main Processing Loop ---

input_filename = 'diverse_rap_corpus_2000.json'
with open(input_filename, 'r', encoding='utf-8') as f:
    songs = json.load(f)

print(f"Loaded {len(songs)} songs. Batch tagging via Google AI Studio...")

output_filename = 'tagged_rap_dataset_gemma4.json'
final_corpus = []
processed_titles = set()

if os.path.exists(output_filename):
    with open(output_filename, 'r', encoding='utf-8') as f:
        final_corpus = json.load(f)
        for song in final_corpus:
            processed_titles.add((song['song'].lower().strip(), song['artist'].lower().strip()))
    print(f"Resuming from {len(final_corpus)} previously tagged songs.")

for i, song in enumerate(songs):
    title_clean = song['title'].lower().strip()
    artist_clean = song['artist'].lower().strip()
    if (title_clean, artist_clean) in processed_titles:
        print(f"[{i+1}/{len(songs)}] Skipping already tagged: {song['title']} by {song['artist']}")
        continue

    print(f"[{i+1}/{len(songs)}] Tagging: {song['title']} by {song['artist']}...")
    
    result = tag_song_with_gemma4(song['title'], song['artist'], song['lyrics'])
    
    if result:
        annotations = result.get("annotations", {})
        fetched_year = annotations.pop("year", None)

        final_song = {
            "artist": song['artist'],
            "song": song['title'],
            "year": fetched_year,
            "lyrics": song['lyrics'],
            "annotations": annotations
        }
        final_corpus.append(final_song)
        processed_titles.add((title_clean, artist_clean))
        
        # Incremental checkpointing
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(final_corpus, f, indent=4)
    else:
        print(f"Skipping {song['title']} due to error.")
    
    # AI Studio free tiers have limits (e.g. 15 RPM). A 4-second safety sleep
    # keeps you clear of concurrency blocks and rate limits.
    time.sleep(4)

print(f"\nSuccessfully finished! Total {len(final_corpus)}")