import os

# Configuration
DATASET = "cub_200_2011"
MODEL = "mobilenetv4_hybrid_medium.ix_e550_r384_in1k"
PLACEMENTS = ["random", "dispersed", "contiguous"]  # order required
VIEWS = 8
SIGMAS = [2.0, 3.0, 4.0, 5.0]
N_LIST = [6, 7, 8, 10]

# a_frac lists per n (rounded to two decimals as requested)
A_FRACS_BY_N = {
    6: [0.10, 0.11, 0.12],
    7: [0.14, 0.15, 0.17],
    8: [0.18, 0.20, 0.22],
    10: [0.28, 0.31, 0.34],
}

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def script_content(code: str, placement: str, n: int, a_frac: float, sigma: float) -> str:
    # Log file matches code name in repo root
    log_name = f"{code}.log"
    return f"""#!/bin/bash
#PBS -P cp23
#PBS -q gpuvolta
#PBS -l ngpus=1            
#PBS -l ncpus=12            
#PBS -l mem=20GB           
#PBS -l walltime=24:30:00  
#PBS -l wd                  
#PBS -l storage=scratch/cp23

module load cuda/12.6.2
# module load python3/3.10.4

source /scratch/cp23/lw4988/PKB/.venv/bin/activate

cd ../..

# Single run generated from grid (no --color-jitter)
python3 train.py --dataset {DATASET} --model {MODEL} --pretrained --augmentation pkb --hflip --rotate --pkb-n {n} --pkb-a-frac {a_frac:.2f} --pkb-sigma {sigma} --pkb-views {VIEWS} --pkb-placement {placement} >> {log_name} 2>&1
"""

def main():
    # Generate kfbxxx scripts in deterministic order
    os.makedirs(THIS_DIR, exist_ok=True)
    submit_lines = ["#!/bin/bash", "set -e", "# Submit all kfb jobs in order"]
    idx = 1
    for placement in PLACEMENTS:  # placement major order
        for n in N_LIST:  # n ascending
            a_fracs = A_FRACS_BY_N[n]
            for a in a_fracs:  # a_frac as specified order
                for sigma in SIGMAS:  # sigma ascending
                    code = f"kfb{idx:03d}"
                    path = os.path.join(THIS_DIR, f"{code}.sh")
                    with open(path, "w", newline="\n") as f:
                        f.write(script_content(code, placement, n, a, sigma))
                    # ensure executable bit on posix systems
                    try:
                        os.chmod(path, 0o755)
                    except Exception:
                        pass
                    submit_lines.append(f"qsub {code}.sh")
                    idx += 1

    # Generate mapping table
    table_path = os.path.join(THIS_DIR, "kfb_table.md")
    with open(table_path, "w", newline="\n") as f:
        f.write("| Code | Placement | n | a_frac | sigma | Val Acc@1 | Val Acc@5 |\n")
        f.write("|------|-----------|---|--------|-------|-----------|-----------|\n")
        idx = 1
        for placement in PLACEMENTS:  # placement major order
            for n in N_LIST:  # n ascending
                a_fracs = A_FRACS_BY_N[n]
                for a in a_fracs:  # a_frac as specified order
                    for sigma in SIGMAS:  # sigma ascending
                        code = f"kfb{idx:03d}"
                        f.write(f"| {code} | {placement} | {n} | {a:.2f} | {sigma} | ? | ? |\n")
                        idx += 1

    # Write submit-all helper
    submit_path = os.path.join(THIS_DIR, "kfb_submit_all.sh")
    with open(submit_path, "w", newline="\n") as f:
        f.write("\n".join(submit_lines) + "\n")
    try:
        os.chmod(submit_path, 0o755)
    except Exception:
        pass

if __name__ == "__main__":
    main()
