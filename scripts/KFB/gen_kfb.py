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


def generate_run_all() -> str:
    """Create a bash script that runs all configs and then parses logs to produce kfb_table.md."""
    lines = ["#!/bin/bash", "set -e"]
    idx = 1
    for placement in PLACEMENTS:
        for n in N_LIST:
            a_fracs = A_FRACS_BY_N[n]
            for a in a_fracs:
                for sigma in SIGMAS:
                    code = f"kfb{idx:03d}"
                    log_file = f"{code}.log"
                    cmd = (
                        f"python3 ../../train.py --dataset {DATASET} --model {MODEL} --pretrained "
                        f"--augmentation pkb --hflip --rotate --pkb-n {n} --pkb-a-frac {a:.2f} "
                        f"--pkb-sigma {sigma} --pkb-views {VIEWS} --pkb-placement {placement} "
                        f"> {log_file} 2>&1"
                    )
                    lines.append(f"echo 'Running {code} ({placement}, n={n}, a={a:.2f}, sigma={sigma})'")
                    lines.append(cmd)
                    idx += 1
    # Add log parsing snippet at the end
    lines.append("")
    lines.append("echo 'Parsing logs for best Val Acc values...'")
    lines.append("for f in kfb*.log; do")
    lines.append("  echo -n \"$f: \";")
    lines.append("  grep 'Val Loss' $f | sed -E 's/.*Val Loss [^|]* T1 ([0-9.]+).*/\\1 &/' | sort -nr | head -1")
    lines.append("done")
    return "\n".join(lines)


def main():
    # Ensure output directory exists
    os.makedirs(THIS_DIR, exist_ok=True)

    run_all_path = os.path.join(THIS_DIR, "run_all.sh")
    with open(run_all_path, "w", newline="\n") as f:
        f.write(generate_run_all())
    try:
        os.chmod(run_all_path, 0o755)
    except Exception:
        pass

    table_path = os.path.join(THIS_DIR, "kfb_table.md")
    with open(table_path, "w", newline="\n") as f:
        f.write("| Code | Placement | n | a_frac | sigma | Val Acc@1 | Val Acc@5 |\n")
        f.write("|------|-----------|---|--------|-------|-----------|-----------|\n")
        idx = 1
        for placement in PLACEMENTS:
            for n in N_LIST:
                a_fracs = A_FRACS_BY_N[n]
                for a in a_fracs:
                    for sigma in SIGMAS:
                        code = f"kfb{idx:03d}"
                        f.write(f"| {code} | {placement} | {n} | {a:.2f} | {sigma} | ? | ? |\n")
                        idx += 1


if __name__ == "__main__":
    main()