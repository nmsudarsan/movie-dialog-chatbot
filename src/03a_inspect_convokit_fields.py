from convokit import Corpus
from pathlib import Path

def main():
    corpus_path = Path.home() / ".convokit" / "saved-corpora" / "movie-corpus"
    corpus = Corpus(filename=str(corpus_path))

    utt = next(corpus.iter_utterances())
    conv = corpus.get_conversation(utt.conversation_id)

    print("Sample utterance id:", utt.id)
    print("utterance.meta keys:", list(utt.meta.keys())[:30])
    print("utterance.meta:", dict(list(utt.meta.items())[:10]))

    print("\nSample conversation id:", conv.id)
    print("conversation.meta keys:", list(conv.meta.keys())[:30])
    print("conversation.meta:", dict(list(conv.meta.items())[:15]))

if __name__ == "__main__":
    main()
