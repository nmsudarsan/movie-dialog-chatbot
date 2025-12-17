from huggingface_hub import snapshot_download
from pathlib import Path

def main():
    out_dir = Path("data/raw_hf")
    out_dir.mkdir(parents=True, exist_ok=True)

    local_path = snapshot_download(
        repo_id="cornell-movie-dialog/cornell_movie_dialog",
        repo_type="dataset",
        local_dir=out_dir,
        local_dir_use_symlinks=False,  
    )

    print("Downloaded to:", local_path)
    print("Files:")
    for p in sorted(Path(local_path).rglob("*")):
        if p.is_file():
            print(" -", p.relative_to(local_path))

if __name__ == "__main__":
    main()
