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
    lines = []
    lines.append("#!/usr/bin/env bash")
    lines.append("set -euo pipefail")
    lines.append("")
    # Move to repo root (two levels up from this script)
    lines.append("# Change to repository root so logs are written there")
    lines.append("cd \"$(dirname \"$0\")\"/../..")
    lines.append("")
    lines.append(f"DATASET=\"{DATASET}\"")
    lines.append(f"MODEL=\"{MODEL}\"")
    lines.append(f"VIEWS={VIEWS}")
    lines.append("MD_PATH=\"scripts/KFB/kfb_table.md\"")
    lines.append("")
    # helper to parse best T1/T5 from a log
    lines.append("get_best_metrics() {")
    lines.append("  local logfile=\"$1\"")
    lines.append("  if [[ ! -f \"$logfile\" ]]; then")
    lines.append("    printf \"0.000 0.000\\n\"")
    lines.append("    return 0")
    lines.append("  fi")
    # awk to scan only the Val section on each epoch line
    lines.append("  awk 'BEGIN{best_t1=0.0;best_t5=0.0} {")
    lines.append("    if (match($0, /Val[^|]*T1[ ]+([0-9.]+)/, m1)) { v=m1[1]+0; if (v>best_t1) best_t1=v }")
    lines.append("    if (match($0, /Val[^|]*T5[ ]+([0-9.]+)/, m2)) { v=m2[1]+0; if (v>best_t5) best_t5=v }")
    lines.append("  } END{ printf \"%.3f %.3f\\n\", best_t1, best_t5 }' \"$logfile\"")
    lines.append("}")
    lines.append("")
    # arrays to hold mapping for table
    lines.append("declare -a codes placements ns afs sigmas")
    lines.append("")
    # runner function
    lines.append("run_one() {")
    lines.append("  local code=\"$1\" placement=\"$2\" n=\"$3\" a=\"$4\" sigma=\"$5\"")
    lines.append("  codes+=(\"$code\"); placements+=(\"$placement\"); ns+=(\"$n\"); afs+=(\"$a\"); sigmas+=(\"$sigma\")")
    lines.append("  echo \"=== Running $code: placement=$placement n=$n a_frac=$a sigma=$sigma ===\"")
    lines.append("  # Tip: activate your environment here if needed, e.g. source .venv/bin/activate")
    lines.append("  ../../python3 train.py \\")
    lines.append("    --dataset \"$DATASET\" --model \"$MODEL\" --pretrained \\")
    lines.append("    --augmentation pkb --hflip --rotate \\")
    lines.append("    --pkb-n \"$n\" --pkb-a-frac \"$a\" --pkb-sigma \"$sigma\" --pkb-views \"$VIEWS\" --pkb-placement \"$placement\" \\")
    lines.append("    >> \"$code.log\" 2>&1")
    lines.append("}")
    lines.append("")
    # enumerate all runs in required order
    idx = 1
    for placement in PLACEMENTS:
        for n in N_LIST:
            a_fracs = A_FRACS_BY_N[n]
            for a in a_fracs:
                for sigma in SIGMAS:
                    code = f"kfb{idx:03d}"
                    lines.append(f"run_one \"{code}\" \"{placement}\" {n} {a:.2f} {sigma}")
                    idx += 1
    lines.append("")
    lines.append("echo \"All runs completed. Building table...\"")
    lines.append("# write header")
    lines.append("cat > \"$MD_PATH\" << 'EOF'\n| Code | Placement | n | a_frac | sigma | Val Acc@1 | Val Acc@5 |\n|------|-----------|---|--------|-------|-----------|-----------|\nEOF")
    lines.append("")
    lines.append("# append each row with best metrics parsed from logs")
    lines.append("for i in \"${!codes[@]}\"; do")
    lines.append("  code=\"${codes[$i]}\"; placement=\"${placements[$i]}\"; n=\"${ns[$i]}\"; a=\"${afs[$i]}\"; sigma=\"${sigmas[$i]}\"")
    lines.append("  read -r best_t1 best_t5 < <(get_best_metrics \"$code.log\")")
    lines.append("  printf '| %s | %s | %s | %.2f | %s | %.3f | %.3f |\n' \"$code\" \"$placement\" \"$n\" \"$a\" \"$sigma\" \"$best_t1\" \"$best_t5\" >> \"$MD_PATH\"")
    lines.append("done")
    lines.append("")
    lines.append("echo \"Wrote table to $MD_PATH\"")
    return "\n".join(lines) + "\n"


def main():
    # Ensure output directory exists
    os.makedirs(THIS_DIR, exist_ok=True)

    # 1) Write run_all.sh
    run_all_path = os.path.join(THIS_DIR, "run_all.sh")
    with open(run_all_path, "w", newline="\n") as f:
        f.write(generate_run_all())
    try:
        os.chmod(run_all_path, 0o755)
    except Exception:
        pass

    # 2) Write an initial mapping table with placeholders (will be overwritten by run_all.sh after runs)
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
