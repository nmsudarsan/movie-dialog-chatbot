import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path
from huggingface_hub import hf_hub_download
import zipfile

# =========================
# Config
# =========================
HF_REPO_ID = "NMSudarsan/movie-dialog-index"   # your HF dataset repo
HF_FILENAME = "index_bundle.zip"              # file inside that repo

INDEX_DIR = Path("data/index")                # where we store extracted index
INDEX_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "meta.parquet"
MODEL_PATH = INDEX_DIR / "model_name.txt"


def ensure_index_present():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    index_path = INDEX_DIR / "faiss.index"
    meta_path = INDEX_DIR / "meta.parquet"
    model_path = INDEX_DIR / "model_name.txt"

    # Already present → nothing to do
    if index_path.exists() and meta_path.exists() and model_path.exists():
        return

    st.info("Downloading prebuilt search index (one-time setup)...")

    zip_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        repo_type="dataset",
    )

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(INDEX_DIR)

    # 🔍 Handle possible nested folder inside ZIP
    nested_dir = INDEX_DIR / "index"
    if nested_dir.exists():
        for f in nested_dir.iterdir():
            f.replace(INDEX_DIR / f.name)
        nested_dir.rmdir()

    # Final validation
    if not (index_path.exists() and meta_path.exists() and model_path.exists()):
        st.error("Downloaded zip, but expected index files were not found after extraction.")
        st.stop()

    st.success("Index ready. Reloading app...")
    st.rerun()



@st.cache_resource
def load_assets():
    """
    Load FAISS index + metadata + embedding model.
    Cached across reruns for performance.
    """
    index = faiss.read_index(str(INDEX_PATH))
    meta = pd.read_parquet(META_PATH)
    model_name = MODEL_PATH.read_text(encoding="utf-8").strip()
    model = SentenceTransformer(model_name)
    return index, meta, model


def search(query: str, k: int = 8) -> pd.DataFrame:
    index, meta, model = load_assets()
    q_emb = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    scores, idxs = index.search(q_emb, k)
    results = meta.iloc[idxs[0]].copy()
    results.insert(0, "score", scores[0])
    return results.sort_values("score", ascending=False)


def summarize_results(query: str, filtered: pd.DataFrame) -> str:
    top_movies = filtered["movie"].head(3).tolist()
    unique_movies = []
    for m in top_movies:
        if m and m not in unique_movies:
            unique_movies.append(m)

    if not unique_movies:
        return f"Your query relates to: **{query}**. I found relevant grounded lines in the dataset."

    movie_str = ", ".join(unique_movies)
    return f"Your query relates to: **{query}**. Top grounded matches come from: **{movie_str}**."


# =========================
# UI
# =========================
st.set_page_config(page_title="Movie Dialog QA Bot", page_icon="🎬", layout="wide")
st.title("🎬 Movie Dialog QA Bot")
st.caption("Ask questions about movie dialog. Results are grounded in the Cornell Movie Dialog corpus (via ConvoKit).")

# ✅ Ensure index exists (downloads from HF if missing)
ensure_index_present()

with st.sidebar:
    st.header("Settings")
    k = st.slider("Number of results", 3, 15, 8)
    min_score = st.slider("Minimum similarity score", 0.0, 1.0, 0.55, 0.01)
    st.markdown("---")
    st.markdown("Tip: try queries like **'who says we're running out of time'** or **'dialog about artificial intelligence'**.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a question about movie dialog...")

if query:
    # show/store user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Retrieve
    results = search(query, k=k)
    filtered = results[results["score"] >= min_score].copy()

    with st.chat_message("assistant"):
        if filtered.empty:
            assistant_md = (
                "I couldn’t find a confident match in the dataset for that question. "
                "Try rephrasing or using a more specific keyword."
            )
            st.markdown(assistant_md)
        else:
            assistant_md = "### Answer (grounded)\n"
            assistant_md += summarize_results(query, filtered) + "\n\n"
            assistant_md += "### Evidence (top matches)\n"

            st.markdown("### Answer (grounded)")
            st.markdown(summarize_results(query, filtered))
            st.markdown("### Evidence (top matches)")

            for _, row in filtered.iterrows():
                label = f"{row['movie']} ({row['year']}) • score {row['score']:.3f}"
                with st.expander(label, expanded=False):
                    speaker_display = row.get("character_name") or row.get("speaker") or row.get("speaker_id") or "Unknown"
                    st.write(f"**Speaker:** {speaker_display}")
                    st.write(f"**Conversation:** {row['conversation_id']}  |  **Utterance:** {row['utterance_id']}")
                    st.markdown(f"> {row['text']}")

    # store assistant response for chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_md})
