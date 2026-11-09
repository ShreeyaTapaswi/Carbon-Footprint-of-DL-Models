"""Train one model on EuroSAT with energy tracking; append one row to results/results.csv.

Example:
    python train.py --model resnet18 --epochs 5 --seed 0
"""
import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from src.data import get_eurosat_loaders
from src.energy import count_flops_params, track_energy
from src.models import MODEL_ZOO, build_model
try:  # private module, not part of the public repo
    from src.score import green_efficiency_score
except ImportError:
    green_efficiency_score = None


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        ps.extend(model(x).argmax(1).cpu().tolist())
        ys.extend(y.tolist())
    return accuracy_score(ys, ps), f1_score(ys, ps, average="macro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet18", choices=list(MODEL_ZOO))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true", help="mixed precision")
    ap.add_argument("--out", default="results/results.csv")
    a = ap.parse_args()

    set_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dl, val_dl, test_dl, n_cls = get_eurosat_loaders(img_size=a.img_size, batch_size=a.batch_size, seed=a.seed)
    model = build_model(a.model, n_cls).to(device)
    gflops, mparams = count_flops_params(model, a.img_size, device)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler(enabled=a.amp and device == "cuda")
    loss_fn = nn.CrossEntropyLoss()

    # ---- training (energy measured) ----
    t0 = time.time()
    with track_energy("train") as tr:
        for ep in range(a.epochs):
            model.train()
            for x, y in train_dl:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device, enabled=a.amp):
                    loss = loss_fn(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
            sched.step()
            _, vf1 = evaluate(model, val_dl, device)
            print(f"epoch {ep + 1}/{a.epochs}  val macro-F1 = {vf1:.4f}")
    train_time = time.time() - t0

    # ---- inference on test set (energy measured separately) ----
    with track_energy("inference") as inf:
        acc, f1 = evaluate(model, test_dl, device)

    total_kwh = tr["energy_kwh"] + inf["energy_kwh"]
    row = {
        "dataset": "eurosat", "model": a.model, "seed": a.seed, "epochs": a.epochs,
        "batch_size": a.batch_size, "amp": a.amp, "img_size": a.img_size,
        "gflops": round(gflops, 3), "params_m": round(mparams, 3),
        "test_acc": round(acc, 4), "test_macro_f1": round(f1, 4),
        "train_time_s": round(train_time, 1),
        "train_energy_kwh": tr["energy_kwh"], "train_co2_kg": tr["co2_kg"],
        "infer_energy_kwh": inf["energy_kwh"], "infer_co2_kg": inf["co2_kg"],
        "total_energy_kwh": total_kwh, "total_co2_kg": tr["co2_kg"] + inf["co2_kg"],
        "ges": round(green_efficiency_score(f1, total_kwh), 4) if green_efficiency_score else None,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if new:
            w.writeheader()
        w.writerow(row)
    print("\n" + "=" * 52)
    print(f" {a.model} | seed {a.seed} | {a.epochs} epochs")
    print(f" Test macro-F1        : {f1:.4f}")
    print(f" Training energy      : {tr['energy_kwh'] * 1000:.2f} Wh")
    print(f" Training CO2 emitted : {tr['co2_kg'] * 1000:.3f} g CO2e")
    print(f" Inference CO2 emitted: {inf['co2_kg'] * 1000:.4f} g CO2e")
    if row["ges"] is not None:
        print(f" Efficiency score     : {row['ges']}")
    print("=" * 52)


if __name__ == "__main__":
    main()
