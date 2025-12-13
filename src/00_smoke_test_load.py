from datasets import load_dataset

def main():
    ds = load_dataset("cornell-movie-dialog/cornell_movie_dialog")
    print(ds)
    print("\nSample row:\n", ds["train"][5])

if __name__ == "__main__":
    main()
