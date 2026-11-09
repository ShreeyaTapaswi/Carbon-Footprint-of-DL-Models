"""Summarise results/results.csv: mean +- std per model, and carbon-footprint charts.

    python summarize.py
Outputs: results/summary.csv, results/carbon_per_model.png, results/accuracy_vs_carbon.png
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("results/results.csv")
if "ges" not in df:
    df["ges"] = float("nan")
df["train_co2_g"] = df["train_co2_kg"] * 1000
df["total_co2_g"] = df["total_co2_kg"] * 1000
df["train_energy_wh"] = df["train_energy_kwh"] * 1000

g = df.groupby(["dataset", "model"]).agg(
    f1_mean=("test_macro_f1", "mean"), f1_std=("test_macro_f1", "std"),
    train_co2_g_mean=("train_co2_g", "mean"), train_co2_g_std=("train_co2_g", "std"),
    train_wh_mean=("train_energy_wh", "mean"),
    gflops=("gflops", "first"), params_m=("params_m", "first"),
    ges_mean=("ges", "mean"), n_runs=("seed", "count"),
).reset_index()
g.to_csv("results/summary.csv", index=False)
print(g.round(4).to_string(index=False))

for ds, d in g.groupby("dataset"):
    d = d.sort_values("train_co2_g_mean")
    plt.figure(figsize=(8, 4.5))
    plt.barh(d["model"], d["train_co2_g_mean"], xerr=d["train_co2_g_std"].fillna(0), color="#4c9f70")
    plt.xlabel("Training CO$_2$ emitted (g CO$_2$e)")
    plt.title(f"Training carbon footprint per model ({ds})")
    plt.tight_layout(); plt.savefig("results/carbon_per_model.png", dpi=200); plt.close()

    plt.figure(figsize=(6.5, 5))
    plt.scatter(d["train_co2_g_mean"], d["f1_mean"], s=60)
    for _, r in d.iterrows():
        plt.annotate(r["model"], (r["train_co2_g_mean"], r["f1_mean"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    plt.xlabel("Training CO$_2$ (g CO$_2$e)"); plt.ylabel("Test macro-F1")
    plt.title(f"Accuracy vs carbon cost ({ds})"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig("results/accuracy_vs_carbon.png", dpi=200); plt.close()
