import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path

INDEX_DIR = Path("data/index")

# --- Basic safety check so the app doesn't crash if index wasn't built yet ---
if not (INDEX_DIR / "faiss.index").exists():
    st.set_page_config(page_title="Movie Dialog QA Bot", page_icon="🎬", layout="wide")
    st.title("🎬 Movie Dialog QA Bot")
    st.error("Vector index not found. Please run: `python src/04_build_vector_index.py`")
    st.stop()

@st.cache_resource
def load_assets():
    index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    meta = pd.read_parquet(INDEX_DIR / "meta.parquet")
    model_name = (INDEX_DIR / "model_name.txt").read_text(encoding="utf-8").strip()
    model = SentenceTransformer(model_name)
    return index, meta, model

def search(query: str, k: int = 8) -> pd.DataFrame:
    index, meta, model = load_assets()
    q_emb = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, idxs = index.search(q_emb, k)
    results = meta.iloc[idxs[0]].copy()
    results.insert(0, "score", scores[0])
    return results

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

# --- UI ---
st.set_page_config(page_title="Movie Dialog QA Bot", page_icon="🎬", layout="wide")
st.title("🎬 Movie Dialog QA Bot")
st.caption("Ask questions about movie dialog. Results are grounded in the Cornell Movie Dialog corpus (via ConvoKit).")

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
    filtered = results[results["score"] >= min_score].sort_values("score", ascending=False)

    response_md = ""

    with st.chat_message("assistant"):
        if filtered.empty:
            response_md = (
                "I couldn’t find a confident match in the dataset for that question. "
                "Try rephrasing or using a more specific keyword."
            )
            st.markdown(response_md)
        else:
            response_md += "### Answer (grounded)\n"
            response_md += summarize_results(query, filtered) + "\n\n"
            response_md += "### Evidence (top matches)\n"

            st.markdown("### Answer (grounded)")
            st.markdown(summarize_results(query, filtered))
            st.markdown("### Evidence (top matches)")

            # Show evidence in expanders (nice UX)
            for _, row in filtered.iterrows():
                label = f"{row['movie']} ({row['year']}) • score {row['score']:.3f}"
                with st.expander(label, expanded=False):
                    st.write(f"**Speaker:** {row['speaker']}")
                    st.write(f"**Conversation:** {row['conversation_id']}  |  **Utterance:** {row['utterance_id']}")
                    st.markdown(f"> {row['text']}")

    # store assistant response (so history is consistent on reruns)
    st.session_state.messages.append({"role": "assistant", "content": response_md})
