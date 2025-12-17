import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path

DATA_PATH = "data/processed/utterances.parquet"
INDEX_DIR = Path("data/index")

def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Load processed utterances
    df = pd.read_parquet(DATA_PATH)
    texts = df["text"].astype(str).tolist()

    # Load embedding model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    print("Encoding utterances...")
    embeddings = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity
    index.add(embeddings)

    # Save index
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))

    # Save metadata needed for lookup
    meta_cols = [
    "utterance_id",
    "conversation_id",
    "speaker",         
    "speaker_id",      
    "character_name",  
    "movie",
    "year",
    "genre",
    "rating",
    "votes",
    "text",
    ]
    df[meta_cols].to_parquet(INDEX_DIR / "meta.parquet", index=False)

    # Save model name for reproducibility
    (INDEX_DIR / "model_name.txt").write_text(model_name, encoding="utf-8")

    print("Done.")
    print("Index size:", index.ntotal)

if __name__ == "__main__":
    main()
