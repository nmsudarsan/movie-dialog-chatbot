from convokit import Corpus, download

def main():
    corpus = Corpus(filename=download("movie-corpus"))
    corpus.print_summary_stats()

if __name__ == "__main__":
    main()
