# split data into training and testing sets
import numpy as np
import pandas as pd
from pathlib import Path

df = pd.read_csv("g2p2k/games2p2k_text.csv")

# split by unique_id
rng = np.random.default_rng(20250926)
unique_ids = df["unique_id"].unique()
rng.shuffle(unique_ids)

n = len(unique_ids)
n_train = int(0.90 * n)
train_ids = set(unique_ids[:n_train])
test_ids  = set(unique_ids[n_train:])

splits = {
    "train": df[df["unique_id"].isin(train_ids)],
    "test":  df[df["unique_id"].isin(test_ids)],
}

# save each split as CSV
for name, sdf in splits.items():
    sdf.to_csv(f"g2p2k/g2p2k_{name}.csv", index=False)
    print("rows:", len(sdf))
