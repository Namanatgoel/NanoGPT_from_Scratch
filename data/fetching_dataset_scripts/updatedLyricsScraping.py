import os
import sys
import json
import time
import random
import re
import lyricsgenius

# Enforce secure environment variables
GENIUS_ACCESS_TOKEN = os.environ.get("GENIUS_ACCESS_TOKEN")
if not GENIUS_ACCESS_TOKEN:
    # Fallback to provided token if environment variable isn't configured yet
    GENIUS_ACCESS_TOKEN = 'Sv3fJoQ1Kwx6K90lGXE6JbFR7hFXAe5BbbnaT-ROjsNogK6h3leEo95y9eaw2Sm1'

genius = lyricsgenius.Genius(
    GENIUS_ACCESS_TOKEN,
    timeout=45,          # Augmented for high-latency mobile endpoints
    retries=3,           # Strategic retry window for connection resets
    sleep_time=1.0       # Standard API request throttling
)

genius.remove_section_headers = True
genius.skip_non_songs = True
genius.verbose = False

# Expanded seed list to guarantee a 2,000 song pool across diverse styles
diverse_artists = [
    # Alternative / Underground
    "MF DOOM", "Earl Sweatshirt", "Tyler, The Creator", "Aesop Rock", "Danny Brown",
    # Trap / Southern
    "Outkast", "Lil Wayne", "Future", "Travis Scott", "Gucci Mane", "Migos"
]

TARGET_TOTAL_SONGS = 500
SONGS_PER_ARTIST = (TARGET_TOTAL_SONGS // len(diverse_artists)) + 15
SAVE_EVERY = 50
OUTPUT_FILENAME = os.path.join('project', 'data', 'diverse_rap_corpus_2000.json')

def clean_genius_artifacts(text):
    """Mechanistic cleanup of low-value scraping artifacts from Genius text strings."""
    if not text:
        return ""
    # Strip initial contributor lines (e.g., "12 Contributors... Lyrics")
    text = re.sub(r'^\d+\s+Contributor.*?\s+Lyrics', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^.*?Lyrics', '', text, count=1, flags=re.IGNORECASE)
    # Strip terminal embed strings (e.g., "24Embed", "143Embed")
    text = re.sub(r'\d*Embed$', '', text)
    # Strip internal promotion markers
    text = re.sub(r'You might also like', '', text, flags=re.IGNORECASE)
    return text.strip()

# Initialize data directories
os.makedirs(os.path.dirname(OUTPUT_FILENAME), exist_ok=True)

rap_corpus = []
seen_titles = set()

# Resume functionality if run gets interrupted halfway
if os.path.exists(OUTPUT_FILENAME):
    try:
        with open(OUTPUT_FILENAME, 'r', encoding='utf-8') as f:
            rap_corpus = json.load(f)
            seen_titles = {song['title'].lower().strip() for song in rap_corpus}
        print(f"[RESUME] Loaded {len(rap_corpus)} existing tracks from checkpoint.")
    except Exception as e:
        print(f"[WARNING] Failed to parse checkpoint, starting fresh: {e}")

print(f"Targeting {TARGET_TOTAL_SONGS} unique tracks (~{SONGS_PER_ARTIST} songs per artist)\n")

for artist_name in diverse_artists:
    if len(rap_corpus) >= TARGET_TOTAL_SONGS:
        break

    print(f"→ Querying Genius Engine: {artist_name}")
    try:
        artist = genius.search_artist(
            artist_name,
            max_songs=SONGS_PER_ARTIST,
            sort="popularity"
        )
        if artist is None:
            print(f"  [SKIP] Artist '{artist_name}' returned null values.")
            continue

        for song in artist.songs:
            if len(rap_corpus) >= TARGET_TOTAL_SONGS:
                break

            clean_title = song.title.lower().strip()
            if clean_title in seen_titles or not song.lyrics:
                continue

            raw_lyrics = song.lyrics
            sanitized_lyrics = clean_genius_artifacts(raw_lyrics)
            
            if not sanitized_lyrics:
                continue

            seen_titles.add(clean_title)
            rap_corpus.append({
                'id': len(rap_corpus) + 1,
                'title': song.title,
                'artist': artist.name,
                'lyrics': sanitized_lyrics
            })
            print(f"  [{len(rap_corpus)}/{TARGET_TOTAL_SONGS}] Extracted: {song.title}")

            if len(rap_corpus) % SAVE_EVERY == 0:
                with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
                    json.dump(rap_corpus, f, indent=2)
                print(f"  ✓ State Checkpoint written safely to disk.")

            # Linear API safety pacing
            time.sleep(random.uniform(1.2, 2.8))

        # Strategic back-off sequence between artist transitions
        cooldown = random.uniform(10, 20)
        print(f"  Finished {artist_name}. Injecting {cooldown:.1f}s cooldown matrix...")
        time.sleep(cooldown)

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Script execution paused by user. Freezing data array state...")
        break
    except Exception as e:
        print(f"  [ERROR] Processing failure for {artist_name}: {e}")
        time.sleep(20) # Dynamic back-off for server drops

# Finalize file save
with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
    json.dump(rap_corpus, f, indent=2)

print(f"\n[COMPLETE] Pipeline execution terminated. Output file: {OUTPUT_FILENAME} (Size: {len(rap_corpus)} elements)")