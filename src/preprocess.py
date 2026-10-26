
from pathlib import Path
import pandas as pd

def preprocess_fields(index_csv, dataset_root, output_dir, manifest_path):
    index = pd.read_csv(index_csv)
    root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
    else:
        manifest = pd.DataFrame(columns=["year", "region", "id", "output", "status", "error"])

    completed = set()
    if not manifest.empty:
        completed = set(
            zip(manifest.loc[manifest["status"] == "ok", "year"].astype(str),
                manifest.loc[manifest["status"] == "ok", "region"].astype(str),
                manifest.loc[manifest["status"] == "ok", "id"].astype(str))
        )

    records = manifest.to_dict("records")

    for _, row in index.iterrows():
        year, region, field_id = str(row["year"]), str(row["region"]).lower(), str(int(row["id"]))
        key = (year, region, field_id)
        out = output_dir / year / region / f"{field_id}.parquet"

        if key in completed and out.exists():
            continue

        source = root / year / "L2A" / region / "csv" / f"{field_id}.csv"
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            if not source.exists():
                raise FileNotFoundError(str(source))

            temporal = pd.read_csv(source)
            temporal.to_parquet(out, index=False)
            records.append({
                "year": year, "region": region, "id": field_id,
                "output": str(out), "status": "ok", "error": ""
            })
            completed.add(key)
        except Exception as exc:
            records.append({
                "year": year, "region": region, "id": field_id,
                "output": str(out), "status": "error", "error": str(exc)
            })

        # Persist progress after every field, so interrupted runs can resume.
        pd.DataFrame(records).drop_duplicates(
            subset=["year", "region", "id"], keep="last"
        ).to_csv(manifest_path, index=False)

    result = pd.DataFrame(records).drop_duplicates(
        subset=["year", "region", "id"], keep="last"
    )
    result.to_csv(manifest_path, index=False)
    return result
