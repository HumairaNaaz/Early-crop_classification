
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import json
from torch.utils.data import Dataset

BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
TEMPORAL_FEATURES = BANDS + ["CP", "day_of_year"]
STATIC_CONT = ["elev", "slope"]
STATIC_CAT = ["sreg_code", "mun_code"]

class CatCropsDataset(Dataset):
    def __init__(self, index_csv, dataset_root, max_steps=144):
        self.index = pd.read_csv(index_csv).reset_index(drop=True)
        self.root = Path(dataset_root)
        self.max_steps = max_steps

        config_dir = Path(__file__).resolve().parents[1] / "config"
        with open(config_dir / "class_mapping.json", encoding="utf-8") as f:
            self.class_mapping = json.load(f)
        with open(config_dir / "categorical_mappings.json", encoding="utf-8") as f:
            self.categorical_mappings = json.load(f)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        row = self.index.iloc[i]
        field_path = (
            self.root / str(int(row["year"])) / "L2A" /
            str(row["region"]).lower() / "csv" / f"{int(row['id'])}.csv"
        )
        temporal = pd.read_csv(field_path)

        dates = pd.to_datetime(temporal["doa"], errors="coerce")
        doy = dates.dt.dayofyear.fillna(1).to_numpy(dtype=np.float32) / 366.0

        x = temporal[BANDS + ["CP"]].to_numpy(dtype=np.float32)
        x = np.concatenate([x, doy[:, None]], axis=1)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        n = min(len(x), self.max_steps)
        features = np.zeros((self.max_steps, len(TEMPORAL_FEATURES)), dtype=np.float32)
        mask = np.zeros(self.max_steps, dtype=bool)
        features[:n] = x[:n]
        mask[:n] = True

        pcrop = pd.to_numeric(pd.Series([row.get("pcrop_code", np.nan)]), errors="coerce").iloc[0]
        current_crop = pd.to_numeric(pd.Series([row.get("CODE_CULTU", np.nan)]), errors="coerce").iloc[0]
        has_previous = pd.notna(pcrop)
        rotation_flag = bool(has_previous and pd.notna(current_crop) and pcrop != current_crop)

        return {
            "temporal": torch.tensor(features, dtype=torch.float32),
            "mask": torch.tensor(mask),
            "static_cont": torch.tensor(
                row[STATIC_CONT].to_numpy(dtype=np.float32), dtype=torch.float32
            ),
            "static_cat": torch.tensor(
                [
                    self.categorical_mappings[col].get(str(row[col]), 0)
                    if pd.notna(row[col]) else 0
                    for col in STATIC_CAT
                ],
                dtype=torch.long
            ),
            "label": torch.tensor(
                int(self.class_mapping[str(int(row["classid"]))]),
                dtype=torch.long
            ),
            "original_label": torch.tensor(int(row["classid"]), dtype=torch.long),
            "previous_crop": torch.tensor(
                int(pcrop) if has_previous else -1, dtype=torch.long
            ),
            "has_previous_crop": torch.tensor(has_previous),
            "rotation_flag": torch.tensor(rotation_flag),
            "field_id": int(row["id"])
        }
