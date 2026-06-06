import os
import json
import time
from groq import Groq

# 1. Environment Verification
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("[CRITICAL] GROQ_API_KEY environment variable missing.")

client = Groq(api_key=GROQ_API_KEY)

# 2. Path Definitions
INPUT_JSON = os.path.join("project", "data", "diverse_rap_corpus_2000.json")
OUTPUT_TXT = os.path.join("project", "data", "literary_rap_conditioned.txt")

def analyze_lyrics(lyrics_text):
    """Queries Groq to classify the lyrics. Enforces strict JSON output."""
    prompt = f"""
    Analyze the following rap lyrics. Output a strict JSON object with exactly two keys: "mood" and "genre".
    Example: {{"mood": "Aggressive", "genre": "Hardcore Rap"}}
    Do not output markdown, explanation, or extra text.
    
    Lyrics: {lyrics_text[:1000]}
    """
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192", # Optimal speed/cost ratio for API rate limits
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"[API ERROR] {e}")
        return {"mood": "Unknown", "genre": "Rap"}

def execute_pipeline():
    print(f"[INITIALIZING] Loading corpus from {INPUT_JSON}")
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
        
    os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)
    
    print(f"[PROCESSING] Commencing tagging and formatting for {len(corpus)} tracks.")
    
    # Open the text file in append mode to stream the formatted syntax directly
    with open(OUTPUT_TXT, 'a', encoding='utf-8') as out_file:
        for idx, song in enumerate(corpus):
            print(f"  → Analyzing [{idx+1}/{len(corpus)}]: {song['title']}")
            
            # Extract metadata
            tags = analyze_lyrics(song['lyrics'])
            mood = tags.get('mood', 'Unknown')
            genre = tags.get('genre', 'Rap')
            
            # Construct the precise Transformer conditioning syntax
            formatted_entry = (
                f"<|startoftext|>\n"
                f"<|mood|> {mood} <|genre|> {genre}\n"
                f"<|bars|>\n"
                f"{song['lyrics']}\n"
                f"<|endoftext|>\n\n"
            )
            
            # Write immediately to disk to prevent RAM bloat or data loss on crash
            out_file.write(formatted_entry)
            
            # CRITICAL: Throttle to respect Groq Free Tier limits (30 Requests/Min)
            time.sleep(2.5) 

    print(f"[COMPLETE] Conditioned text dataset compiled at: {OUTPUT_TXT}")

if __name__ == "__main__":
    if not os.path.exists(INPUT_JSON):
        print(f"[FATAL] Input file not found: {INPUT_JSON}. Run the Genius scraper first.")
        exit(1)
    
    # Install groq if missing: pip install groq
    execute_pipeline()