# import pandas as pd
# from convokit import Corpus, download

# def main():
#     corpus = Corpus(filename=download("movie-corpus"))

#     rows = []

#     for utt in corpus.iter_utterances():
#         rows.append({
#             "utterance_id": utt.id,
#             "conversation_id": utt.conversation_id,
#             "speaker": utt.speaker.id if utt.speaker else None,
#             "movie": utt.meta.get("movie_name"),
#             "year": utt.meta.get("release_year"),
#             "text": utt.text
#         })

#     df = pd.DataFrame(rows)

#     print("Total rows:", len(df))
#     print(df.head())

#     # save processed data
#     output_path = "data/processed/utterances.parquet"
#     df.to_parquet(output_path, index=False)

#     print(f"\nSaved to {output_path}")

# if __name__ == "__main__":
#     main()


import pandas as pd
from convokit import Corpus
from pathlib import Path

def main():
    corpus_path = Path.home() / ".convokit" / "saved-corpora" / "movie-corpus"
    corpus = Corpus(filename=str(corpus_path))

    # Build conversation -> movie/year lookup
    conv_rows = []
    for conv in corpus.iter_conversations():
        conv_rows.append({
            "conversation_id": conv.id,
            "movie": conv.meta.get("movie_name") or conv.meta.get("movie_title"),
            "year": conv.meta.get("release_year") or conv.meta.get("movie_year"),
            "genre": conv.meta.get("genre") or conv.meta.get("movie_genres"),
            "rating": conv.meta.get("rating") or conv.meta.get("movieIMDBRating"),
            "votes": conv.meta.get("votes") or conv.meta.get("movieNoIMDBVotes"),
        })
    conv_df = pd.DataFrame(conv_rows)

    # Build utterances table
    utt_rows = []
    for utt in corpus.iter_utterances():
        utt_rows.append({
            "utterance_id": utt.id,
            "conversation_id": utt.conversation_id,
            "speaker": utt.speaker.id if utt.speaker else None,
            "text": utt.text,
        })
    utt_df = pd.DataFrame(utt_rows)

    # Join to attach movie metadata
    df = utt_df.merge(conv_df, on="conversation_id", how="left")

    print("Total rows:", len(df))
    print("Missing movie:", df["movie"].isna().sum())
    print(df.head())

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    out_path = "data/processed/utterances.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
