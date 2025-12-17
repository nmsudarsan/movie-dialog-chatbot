# 🎬 Movie Dialog QA Bot (Semantic Search)

A Streamlit app that performs **semantic search** over the Cornell Movie-Dialogs Corpus (via ConvoKit).  
Users ask a question, and the app retrieves the most semantically similar movie dialog lines and displays them as **grounded evidence** (retrieval-only, no LLM generation).

## 🚀 Live app
- **Streamlit:** https://movie-dialog-chatbot.streamlit.app/

## 📦 Prebuilt index (for deployment)
- **Hugging Face dataset (FAISS + metadata):**  
  https://huggingface.co/datasets/NMSudarsan/movie-dialog-index

## 🎯 Demo
- Query: `we’re running out of time`
- Returns top matching dialog lines with movie title/year and IDs.

## 🧠 How it works
1. **Data source:** ConvoKit `movie-corpus` (Cornell Movie-Dialogs Corpus)
2. **Embeddings:** SentenceTransformer `all-MiniLM-L6-v2`
3. **Search:** FAISS index for fast nearest-neighbor retrieval
4. **UI:** Streamlit chat interface with evidence expanders
5. **Deployment:** App auto-downloads a prebuilt index from Hugging Face on first run

## 📁 Project structure
```text
movie-dialog-chatbot/
├── app.py
├── requirements.txt
├── runtime.txt
├── README.md
├── .gitignore
├── src/                         # scripts used to build the dataset/index locally
│   ├── 02_download_with_convokit.py
│   ├── 03a_inspect_convokit_fields.py
│   ├── 03_build_utterances_table.py
│   ├── 04_build_vector_index.py
│   └── 05_search_test.py
└── data/
    ├── processed/               # generated locally (optional)
    └── index/                   # placeholders only; real index is downloaded from HF

```

## Setup (local)
python -m venv venv
# activate venv
pip install -r requirements.txt
python -m streamlit run app.py

## Developer: rebuild the index locally (optional)
If you want to regenerate the FAISS index yourself:

- python src/02_download_with_convokit.py
- python src/03_build_utterances_table.py
- python src/04_build_vector_index.py

## Notes

This app is retrieval-only and shows grounded evidence from the dataset.

Large artifacts (FAISS index) are not committed to GitHub.

The production app uses a prebuilt index hosted on Hugging Face for fast, reliable startup.
