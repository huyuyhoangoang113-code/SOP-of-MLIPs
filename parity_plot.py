#!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from ase.io import read

from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core import FAIRChemCalculator


# ======================
# PATHS
# ======================
OUTCAR_PATH = "/home/hoang0000/uma/NMC_new/train/OUTCAR/1/OUTCAR"
#CHECKPOINT_PATH = "/home/hoang0000/uma-s-1p2.pt"
CHECKPOINT_PATH = "/home/hoang0000/uma_finetune_runs/NMC_run_20260107/nmc_finetune/checkpoints/final/inference_ckpt.pt"
OUTPUT_IMAGE = "force_parity.png"


# ======================
# DEVICE
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(1)
torch.set_grad_enabled(False)


# ======================
# LOAD ML MODEL
# ======================
print("Loading ML model...")

predictor = load_predict_unit(CHECKPOINT_PATH, device=DEVICE)
calc = FAIRChemCalculator(predictor, task_name="omat")

print("Model loaded")


# ======================
# READ OUTCAR (SAFE ASE METHOD)
# ======================
print("Reading OUTCAR...")

try:
    frames = read(OUTCAR_PATH, format="vasp-out", index=":")
except Exception as e:
    print("ASE read failed, fallback to default reader:", e)
    frames = read(OUTCAR_PATH, index=":")

print(f"Total frames: {len(frames)}")


# ======================
# FILTER VALID FRAMES
# ======================
valid_frames = []

for i, atoms in enumerate(frames):
    try:
        _ = atoms.get_positions()
        _ = atoms.get_forces()
        valid_frames.append(atoms)
    except Exception as e:
        print(f"Skip step {i}: {e}")

frames = valid_frames

print(f"Valid frames: {len(frames)}")

if len(frames) == 0:
    raise RuntimeError("No valid ionic steps found")


# ======================
# COLLECT FORCES
# ======================
dft_all = []
ml_all = []

print("Running ML inference...")

for i, atoms in enumerate(frames):
    try:
        f_dft = atoms.get_forces()

        atoms.calc = calc
        f_ml = atoms.get_forces()

        dft_all.append(f_dft.reshape(-1))
        ml_all.append(f_ml.reshape(-1))

    except Exception as e:
        print(f"Skip step {i}: {e}")


if len(dft_all) == 0:
    raise RuntimeError("No force data collected")


y_true = np.concatenate(dft_all)
y_pred = np.concatenate(ml_all)


# ======================
# METRICS
# ======================
mae = np.mean(np.abs(y_true - y_pred))
rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

print("\n===== RESULTS =====")
print(f"MAE  = {mae:.6f}")
print(f"RMSE = {rmse:.6f}")
print(f"N    = {len(y_true)}")


# ======================
# PARITY PLOT
# ======================
plt.figure(figsize=(8, 8), dpi=300)

limit = max(np.max(np.abs(y_true)), np.max(np.abs(y_pred))) * 1.05

hb = plt.hexbin(
    y_true,
    y_pred,
    gridsize=220,
    bins="log",
    mincnt=1,
    cmap="viridis"
)

plt.colorbar(hb, label="log10(count)")

plt.plot([-limit, limit], [-limit, limit], "r--", linewidth=1)

plt.xlim(-limit, limit)
plt.ylim(-limit, limit)

plt.xlabel("DFT force (eV/A)")
plt.ylabel("ML force (eV/A)")
plt.title("Force parity: DFT vs MLIP")

plt.text(
    0.05,
    0.95,
    f"MAE = {mae:.4f}\nRMSE = {rmse:.4f}\nPoints = {len(y_true)}",
    transform=plt.gca().transAxes,
    bbox=dict(facecolor="white", alpha=0.6),
    verticalalignment="top"
)

plt.tight_layout()
plt.savefig(OUTPUT_IMAGE)

print(f"Saved plot: {OUTPUT_IMAGE}")
