# 🎬 Movie Dialog QA Bot (Semantic Search)

A Streamlit app that performs **semantic search** over the Cornell Movie-Dialogs Corpus (via ConvoKit).  
Users ask a question, the app retrieves the most semantically similar movie dialog lines and displays them as grounded evidence.

## Demo
- Query: “we’re running out of time”
- Returns top matching dialog lines with movie title/year and IDs.

## How it works
1. **Data source:** ConvoKit `movie-corpus` (Cornell Movie-Dialogs Corpus)
2. **Processing:** Convert corpus into a flat table (one row per utterance) and join movie metadata from conversations
3. **Embeddings:** SentenceTransformer `all-MiniLM-L6-v2`
4. **Search:** FAISS index for fast nearest-neighbor retrieval
5. **UI:** Streamlit chat interface with evidence expanders

## Project structure
```text
movie-dialog-chatbot/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── 02_download_with_convokit.py
│   ├── 03a_inspect_convokit_fields.py
│   ├── 03_build_utterances_table.py
│   ├── 04_build_vector_index.py
│   └── 05_search_test.py
└── data/
    ├── processed/   (generated locally)
    └── index/       (generated locally)

```

## Setup (local)
python -m venv venv
# activate venv
pip install -r requirements.txt

## Build the index (local)

Run these once:

- python src/02_download_with_convokit.py
- python src/03_build_utterances_table.py
- python src/04_build_vector_index.py

## Run the app
python -m streamlit run app.py

