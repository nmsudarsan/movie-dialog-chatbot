import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path

INDEX_DIR = Path("data/index")

def load_assets():
    index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    meta = pd.read_parquet(INDEX_DIR / "meta.parquet")
    model_name = (INDEX_DIR / "model_name.txt").read_text(encoding="utf-8").strip()
    model = SentenceTransformer(model_name)
    return index, meta, model

def search(query: str, k: int = 8):
    index, meta, model = load_assets()

    q_emb = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, idxs = index.search(q_emb, k)

    results = meta.iloc[idxs[0]].copy()
    results.insert(0, "score", scores[0])
    return results

def main():
    queries = [
        "We're running out of time",
        "two characters argue about loyalty",
        "dialog about artificial intelligence",
        "a character apologizes reluctantly",
    ]

    for q in queries:
        print("\n" + "="*90)
        print("QUERY:", q)
        res = search(q, k=8)

        for i, row in res.iterrows():
            print(f"\nScore: {row['score']:.3f}")
            print(f"{row['movie']} ({row['year']}) | speaker={row['speaker']} | conv={row['conversation_id']} | utt={row['utterance_id']}")
            print("Text:", row["text"])

if __name__ == "__main__":
    main()
