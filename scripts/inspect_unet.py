from safetensors import safe_open
import pandas as pd
import os

MODEL_PATH = "models/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"

os.makedirs("reports", exist_ok=True)

rows = []

with safe_open(MODEL_PATH, framework="pt", device="cpu") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        shape = list(tensor.shape)

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

summary = (
    df.assign(component=df["tensor"].str.split(".").str[0])
      .groupby("component")["parameters"]
      .sum()
      .reset_index()
)

summary.to_csv("reports/summary.csv", index=False)

print("\n=== COMPONENT SUMMARY ===")
print(summary)
print("\n=== FIRST 20 TENSORS ===")
print(df.head(20))
