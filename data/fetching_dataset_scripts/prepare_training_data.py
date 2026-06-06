import pandas as pd
import json

def flatten_csv_to_jsonl(input_csv: str, output_jsonl: str):
    """
    Transforms tabular CSV data into a flattened, heavily conditioned 
    text sequence mapped to a JSONL format for LLM fine-tuning.
    """
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        raise RuntimeError(f"Input file {input_csv} not found.")

    processed_count = 0

    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            # Extract and sanitize metadata
            artist = str(row['artist']).strip() if pd.notna(row['artist']) else "Unknown"
            song = str(row['song']).strip() if pd.notna(row['song']) else "Unknown"
            
            # Handle float casting issues for missing years in the dataset
            year = str(int(row['year'])) if pd.notna(row['year']) else "Unknown"
            
            mood = str(row['mood']).strip() if pd.notna(row['mood']) else "Unknown"
            genre = str(row['genre']).strip() if pd.notna(row['genre']) else "Unknown"
            rhyme = str(row['rhyme_scheme']).strip() if pd.notna(row['rhyme_scheme']) else "Unknown"
            cadence = str(row['cadence']).strip() if pd.notna(row['cadence']) else "Unknown"
            lyrics = str(row['lyrics']).strip() if pd.notna(row['lyrics']) else ""

            # Discard corrupted rows with no lyrics
            if not lyrics:
                continue

            # Construct the explicitly conditioned string sequence
            text_sequence = (
                f"<|startoftext|>\n"
                f"<|artist|> {artist} <|song|> {song} <|year|> {year}\n"
                f"<|mood|> {mood} <|genre|> {genre} <|rhyme_scheme|> {rhyme} <|cadence|> {cadence}\n"
                f"<|bars|>\n"
                f"{lyrics}\n"
                f"<|endoftext|>"
            )

            # Package into standard JSONL structure
            json_record = {"text": text_sequence}
            f.write(json.dumps(json_record, ensure_ascii=False) + '\n')
            processed_count += 1

    print(f"Data flattening complete. {processed_count} sequences written to {output_jsonl}.")

if __name__ == "__main__":
    INPUT_CSV = "cleaned_combined.csv"
    OUTPUT_JSONL = "training_corpus.jsonl"
    flatten_csv_to_jsonl(INPUT_CSV, OUTPUT_JSONL)