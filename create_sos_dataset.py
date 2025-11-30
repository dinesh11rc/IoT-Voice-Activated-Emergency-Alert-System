# create_sos_dataset.py
import numpy as np
import pandas as pd
import random

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

LABELS = ["help", "emergency", "sos"]
N_SAMPLES_PER_LABEL = 400
N_FEATURES = 13  # typical MFCC count

def base_vector(label):
    if label == "help":
        return np.array([-105, 22, 15, 10, 3, -1.5, -2, 0, 1, 2, 0.5, -1.2, 0.9])
    if label == "emergency":
        return np.array([-98, 30, 20, 8, 2, -0.5, -1, 1, 0, 1.2, 0.3, -0.7, 1.5])
    if label == "sos":
        return np.array([-110, 18, 12, 9, 4, -2.0, -1.8, -0.5, 0.5, 1.6, 0.7, -1.0, 0.2])
    return np.zeros(N_FEATURES)

rows = []
for label in LABELS:
    base = base_vector(label)
    for i in range(N_SAMPLES_PER_LABEL):
        noise = np.random.normal(scale=2.0, size=N_FEATURES)  # variability
        sample = base + noise
        rows.append(list(sample) + [label])

cols = [f"mfcc_{i+1}" for i in range(N_FEATURES)] + ["label"]
df = pd.DataFrame(rows, columns=cols)
df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
df.to_csv("sos_dataset.csv", index=False)
print("Saved sos_dataset.csv — rows:", len(df))
