#!/usr/bin/env python3
import os
import argparse

# ASE imports
from ase.io import read as ase_read
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase import units

# FAIRChem imports
from fairchem.core import pretrained_mlip, FAIRChemCalculator

# --- Settings ---
INPUT = next(
    (f for f in os.listdir(".") 
     if f in ["POSCAR"] or f.lower().endswith((".vasp", ".cif", ".xyz"))), 
    None
) # accept .vasp, .cif and .xyz
TIME_STEP_FS = 2.0 # more accurate but more computational cost with smaller value
LOG_INTERVAL = 50 # save frame each 50 time steps

def run_md_simulation(temperature_k, total_time_ps):
    total_steps = int(total_time_ps * 1000 / TIME_STEP_FS)

    traj_file = f"md_{temperature_k}K.traj" 
    log_file = f"md_{temperature_k}K.log"

    print("--- Simulation Protocol ---")
    print(f"Total: {total_time_ps:4d} ps ({total_steps:7d} steps)")
    print("-------------------------")

    steps_done = 0
    atoms = None
    if os.path.exists(traj_file) and os.path.getsize(traj_file) > 0:
        try:
            frames = ase_read(traj_file, index=":")
            if frames:
                steps_done = len(frames)
                atoms = frames[-1]
                print(f"Restarting from existing trajectory: {steps_done}/{total_steps} steps completed.")
        except Exception as e:
            print(f"Warning: Failed to read existing trajectory '{traj_file}': {e}")
            if os.path.exists(traj_file):
                os.rename(traj_file, f"{traj_file}.bak")
                print(f"Moved corrupted file to {traj_file}.bak")

    if atoms is None:
        print(f"Loading initial structure from '{INPUT}'...")
        atoms = ase_read(INPUT)

    steps_left = total_steps - steps_done
    if steps_left <= 0:
        print(f"Simulation for {temperature_k}K already complete.")
        return

    print(f"Running {steps_left} more steps at {temperature_k}K...")

    # ===== FINAL, SIMPLIFIED CODE BLOCK =====
    print("Loading FAIRChem UMA model from local cache...")
    calc = FAIRChemCalculator.from_model_checkpoint(
    name_or_path="/home/hoang0000/uma/checkpoints/uma-s-1p2.pt", 
    task_name="omat",
    device="cuda"
    )

    atoms.calc = calc

    dyn = Langevin(
        atoms,
        timestep=TIME_STEP_FS * units.fs,
        temperature_K=temperature_k,
        friction=0.01,
        logfile=log_file,
        loginterval=LOG_INTERVAL
    )

    traj = Trajectory(traj_file, "a", atoms)
    dyn.attach(traj.write, interval=LOG_INTERVAL)

    dyn.run(steps_left)

    print(f"Finished MD at {temperature_k}K.")

# --- Main function ---
def main():
    parser = argparse.ArgumentParser(
        description="Run FAIRChem MD for a single temperature with a simplified total-time protocol."
    )
    parser.add_argument("temperature", type=int, help="Target temperature in Kelvin")
    parser.add_argument("total_time_ps", type=int, help="Total simulation time in ps")
    args = parser.parse_args()
    run_md_simulation(args.temperature, args.total_time_ps)

if __name__ == "__main__":
    main()
