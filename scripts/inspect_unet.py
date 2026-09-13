
from safetensors import safe_open
import pandas as pd
import os

MODEL_PATH = "JuggernautXL.safetensors"
os.makedirs("reports", exist_ok=True)

rows = []

with safe_open(MODEL_PATH, framework="pt", device="cpu") as f:
    for key in f.keys():
        shape = f.get_tensor(key).shape
        params = 1
        for s in shape:
            params *= s

        rows.append({
            "tensor": key,
            "shape": str(shape),
            "parameters": params
        })

df = pd.DataFrame(rows)
df.to_csv("reports/layers.csv", index=False)

summary = df.groupby(df["tensor"].str.split(".").str[0]).sum(numeric_only=True)
summary.to_csv("reports/summary.csv")

print(df.head())
print(summary)
