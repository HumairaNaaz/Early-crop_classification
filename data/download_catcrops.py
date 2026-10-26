from pathlib import Path
import kagglehub

def download_catcrops():
    path = Path(kagglehub.dataset_download("irtaremotesensing/catcrops-dataset"))
    candidates = [path / "catcrops_dataset", path]
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "2022" / "L2A" / "lleida.csv").exists():
            return candidate
    raise FileNotFoundError(f"CatCrops index files not found under {path}")

if __name__ == "__main__":
    print(download_catcrops())
