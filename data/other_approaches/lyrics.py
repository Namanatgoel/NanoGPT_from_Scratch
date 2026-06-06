import lyricsgenius
import json
import time
import random

GENIUS_ACCESS_TOKEN = 'Sv3fJoQ1Kwx6K90lGXE6JbFR7hFXAe5BbbnaT-ROjsNogK6h3leEo95y9eaw2Sm1'

genius = lyricsgenius.Genius(
    GENIUS_ACCESS_TOKEN,
    timeout=30,          # increased — lyrics fetches are slow
    retries=0,           # don't retry; handle errors manually
    sleep_time=0.5       # built-in per-request delay
)
genius.remove_section_headers = True
genius.skip_non_songs = True
genius.verbose = False   # reduces noise and minor overhead

diverse_artists = [
   # "Nas", "Tupac", "The Notorious B.I.G.", "Wu-Tang Clan", "Rakim", "Jay-Z",
   # "Kendrick Lamar", "J. Cole", "Mos Def", "Common", "Lupe Fiasco", "Black Thought",
    "MF DOOM", "Earl Sweatshirt", "Tyler, The Creator", "Aesop Rock", "Danny Brown"
    "Outkast", "Lil Wayne", "Future", "Travis Scott", "Gucci Mane", "Migos",
    # "Drake", "Kanye West", "Eminem", "Post Malone", "21 Savage",
    # "Missy Elliott", "Lauryn Hill", "Nicki Minaj", "Megan Thee Stallion", "Little Simz",
    # "Dave", "Stormzy", "Skepta", "Central Cee",
    # "Karan Aujla", "DIVINE", "Sidhu Moose Wala", "KR$NA", "Seedhe Maut",
    # "Death Grips", "Denzel Curry", "Run The Jewels", "JPEGMAFIA", "Vince Staples"
]

TARGET_TOTAL_SONGS = 500
SONGS_PER_ARTIST = (TARGET_TOTAL_SONGS // len(diverse_artists)) + 5
SAVE_EVERY = 100         # checkpoint saves so a crash doesn't lose everything

rap_corpus = []
seen_titles = set()
output_filename = 'diverse_rap_corpus_2000.json'

print(f"Targeting {TARGET_TOTAL_SONGS} songs, ~{SONGS_PER_ARTIST} per artist\n")

for artist_name in diverse_artists:
    if len(rap_corpus) >= TARGET_TOTAL_SONGS:
        break

    print(f"\n→ Fetching: {artist_name}")
    try:
        artist = genius.search_artist(
            artist_name,
            max_songs=SONGS_PER_ARTIST,
            sort="popularity"
        )
        if artist is None:
            print(f"  Skipped — artist not found")
            continue

        for song in artist.songs:
            if len(rap_corpus) >= TARGET_TOTAL_SONGS:
                break

            clean_title = song.title.lower().strip()
            if clean_title in seen_titles or not song.lyrics:
                continue

            seen_titles.add(clean_title)
            rap_corpus.append({
                'id': len(rap_corpus) + 1,
                'title': song.title,
                'artist': artist.name,
                'lyrics': song.lyrics
            })
            print(f"  [{len(rap_corpus)}/{TARGET_TOTAL_SONGS}] {song.title}")

            # Checkpoint save
            if len(rap_corpus) % SAVE_EVERY == 0:
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(rap_corpus, f, indent=2)
                print(f"  ✓ Checkpoint saved at {len(rap_corpus)} songs")

            # Randomized per-song delay — looks more human, avoids burst detection
            time.sleep(random.uniform(1.0, 2.5))

        # Longer cooldown between artists
        cooldown = random.uniform(8, 15)
        print(f"  Artist done. Cooling down {cooldown:.1f}s...")
        time.sleep(cooldown)

    except KeyboardInterrupt:
        print("\nInterrupted by user — saving progress...")
        break
    except Exception as e:
        print(f"  Error on {artist_name}: {e}")
        time.sleep(random.uniform(10, 20))  # back off on errors

# Final save
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(rap_corpus, f, indent=2)

print(f"\nDone. Saved {len(rap_corpus)} songs to {output_filename}")