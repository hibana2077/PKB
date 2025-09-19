#!/usr/bin/env bash
set -euo pipefail

# Change to repository root so logs are written there
cd "$(dirname "$0")"/../..

DATASET="cub_200_2011"
MODEL="mobilenetv4_hybrid_medium.ix_e550_r384_in1k"
VIEWS=8
MD_PATH="scripts/KFB/kfb_table.md"

get_best_metrics() {
  local logfile="$1"
  if [[ ! -f "$logfile" ]]; then
    printf "0.000 0.000\n"
    return 0
  fi
  awk 'BEGIN{best_t1=0.0;best_t5=0.0} {
    if (match($0, /Val[^|]*T1[ ]+([0-9.]+)/, m1)) { v=m1[1]+0; if (v>best_t1) best_t1=v }
    if (match($0, /Val[^|]*T5[ ]+([0-9.]+)/, m2)) { v=m2[1]+0; if (v>best_t5) best_t5=v }
  } END{ printf "%.3f %.3f\n", best_t1, best_t5 }' "$logfile"
}

declare -a codes placements ns afs sigmas

run_one() {
  local code="$1" placement="$2" n="$3" a="$4" sigma="$5"
  codes+=("$code"); placements+=("$placement"); ns+=("$n"); afs+=("$a"); sigmas+=("$sigma")
  echo "=== Running $code: placement=$placement n=$n a_frac=$a sigma=$sigma ==="
  # Tip: activate your environment here if needed, e.g. source .venv/bin/activate
  python3 ../../train.py \
    --dataset "$DATASET" --model "$MODEL" --pretrained \
    --augmentation pkb --hflip --rotate \
    --pkb-n "$n" --pkb-a-frac "$a" --pkb-sigma "$sigma" --pkb-views "$VIEWS" --pkb-placement "$placement" \
    >> "$code.log" 2>&1
}

run_one "kfb001" "random" 6 0.10 2.0
run_one "kfb002" "random" 6 0.10 3.0
run_one "kfb003" "random" 6 0.10 4.0
run_one "kfb004" "random" 6 0.10 5.0
run_one "kfb005" "random" 6 0.11 2.0
run_one "kfb006" "random" 6 0.11 3.0
run_one "kfb007" "random" 6 0.11 4.0
run_one "kfb008" "random" 6 0.11 5.0
run_one "kfb009" "random" 6 0.12 2.0
run_one "kfb010" "random" 6 0.12 3.0
run_one "kfb011" "random" 6 0.12 4.0
run_one "kfb012" "random" 6 0.12 5.0
run_one "kfb013" "random" 7 0.14 2.0
run_one "kfb014" "random" 7 0.14 3.0
run_one "kfb015" "random" 7 0.14 4.0
run_one "kfb016" "random" 7 0.14 5.0
run_one "kfb017" "random" 7 0.15 2.0
run_one "kfb018" "random" 7 0.15 3.0
run_one "kfb019" "random" 7 0.15 4.0
run_one "kfb020" "random" 7 0.15 5.0
run_one "kfb021" "random" 7 0.17 2.0
run_one "kfb022" "random" 7 0.17 3.0
run_one "kfb023" "random" 7 0.17 4.0
run_one "kfb024" "random" 7 0.17 5.0
run_one "kfb025" "random" 8 0.18 2.0
run_one "kfb026" "random" 8 0.18 3.0
run_one "kfb027" "random" 8 0.18 4.0
run_one "kfb028" "random" 8 0.18 5.0
run_one "kfb029" "random" 8 0.20 2.0
run_one "kfb030" "random" 8 0.20 3.0
run_one "kfb031" "random" 8 0.20 4.0
run_one "kfb032" "random" 8 0.20 5.0
run_one "kfb033" "random" 8 0.22 2.0
run_one "kfb034" "random" 8 0.22 3.0
run_one "kfb035" "random" 8 0.22 4.0
run_one "kfb036" "random" 8 0.22 5.0
run_one "kfb037" "random" 10 0.28 2.0
run_one "kfb038" "random" 10 0.28 3.0
run_one "kfb039" "random" 10 0.28 4.0
run_one "kfb040" "random" 10 0.28 5.0
run_one "kfb041" "random" 10 0.31 2.0
run_one "kfb042" "random" 10 0.31 3.0
run_one "kfb043" "random" 10 0.31 4.0
run_one "kfb044" "random" 10 0.31 5.0
run_one "kfb045" "random" 10 0.34 2.0
run_one "kfb046" "random" 10 0.34 3.0
run_one "kfb047" "random" 10 0.34 4.0
run_one "kfb048" "random" 10 0.34 5.0
run_one "kfb049" "dispersed" 6 0.10 2.0
run_one "kfb050" "dispersed" 6 0.10 3.0
run_one "kfb051" "dispersed" 6 0.10 4.0
run_one "kfb052" "dispersed" 6 0.10 5.0
run_one "kfb053" "dispersed" 6 0.11 2.0
run_one "kfb054" "dispersed" 6 0.11 3.0
run_one "kfb055" "dispersed" 6 0.11 4.0
run_one "kfb056" "dispersed" 6 0.11 5.0
run_one "kfb057" "dispersed" 6 0.12 2.0
run_one "kfb058" "dispersed" 6 0.12 3.0
run_one "kfb059" "dispersed" 6 0.12 4.0
run_one "kfb060" "dispersed" 6 0.12 5.0
run_one "kfb061" "dispersed" 7 0.14 2.0
run_one "kfb062" "dispersed" 7 0.14 3.0
run_one "kfb063" "dispersed" 7 0.14 4.0
run_one "kfb064" "dispersed" 7 0.14 5.0
run_one "kfb065" "dispersed" 7 0.15 2.0
run_one "kfb066" "dispersed" 7 0.15 3.0
run_one "kfb067" "dispersed" 7 0.15 4.0
run_one "kfb068" "dispersed" 7 0.15 5.0
run_one "kfb069" "dispersed" 7 0.17 2.0
run_one "kfb070" "dispersed" 7 0.17 3.0
run_one "kfb071" "dispersed" 7 0.17 4.0
run_one "kfb072" "dispersed" 7 0.17 5.0
run_one "kfb073" "dispersed" 8 0.18 2.0
run_one "kfb074" "dispersed" 8 0.18 3.0
run_one "kfb075" "dispersed" 8 0.18 4.0
run_one "kfb076" "dispersed" 8 0.18 5.0
run_one "kfb077" "dispersed" 8 0.20 2.0
run_one "kfb078" "dispersed" 8 0.20 3.0
run_one "kfb079" "dispersed" 8 0.20 4.0
run_one "kfb080" "dispersed" 8 0.20 5.0
run_one "kfb081" "dispersed" 8 0.22 2.0
run_one "kfb082" "dispersed" 8 0.22 3.0
run_one "kfb083" "dispersed" 8 0.22 4.0
run_one "kfb084" "dispersed" 8 0.22 5.0
run_one "kfb085" "dispersed" 10 0.28 2.0
run_one "kfb086" "dispersed" 10 0.28 3.0
run_one "kfb087" "dispersed" 10 0.28 4.0
run_one "kfb088" "dispersed" 10 0.28 5.0
run_one "kfb089" "dispersed" 10 0.31 2.0
run_one "kfb090" "dispersed" 10 0.31 3.0
run_one "kfb091" "dispersed" 10 0.31 4.0
run_one "kfb092" "dispersed" 10 0.31 5.0
run_one "kfb093" "dispersed" 10 0.34 2.0
run_one "kfb094" "dispersed" 10 0.34 3.0
run_one "kfb095" "dispersed" 10 0.34 4.0
run_one "kfb096" "dispersed" 10 0.34 5.0
run_one "kfb097" "contiguous" 6 0.10 2.0
run_one "kfb098" "contiguous" 6 0.10 3.0
run_one "kfb099" "contiguous" 6 0.10 4.0
run_one "kfb100" "contiguous" 6 0.10 5.0
run_one "kfb101" "contiguous" 6 0.11 2.0
run_one "kfb102" "contiguous" 6 0.11 3.0
run_one "kfb103" "contiguous" 6 0.11 4.0
run_one "kfb104" "contiguous" 6 0.11 5.0
run_one "kfb105" "contiguous" 6 0.12 2.0
run_one "kfb106" "contiguous" 6 0.12 3.0
run_one "kfb107" "contiguous" 6 0.12 4.0
run_one "kfb108" "contiguous" 6 0.12 5.0
run_one "kfb109" "contiguous" 7 0.14 2.0
run_one "kfb110" "contiguous" 7 0.14 3.0
run_one "kfb111" "contiguous" 7 0.14 4.0
run_one "kfb112" "contiguous" 7 0.14 5.0
run_one "kfb113" "contiguous" 7 0.15 2.0
run_one "kfb114" "contiguous" 7 0.15 3.0
run_one "kfb115" "contiguous" 7 0.15 4.0
run_one "kfb116" "contiguous" 7 0.15 5.0
run_one "kfb117" "contiguous" 7 0.17 2.0
run_one "kfb118" "contiguous" 7 0.17 3.0
run_one "kfb119" "contiguous" 7 0.17 4.0
run_one "kfb120" "contiguous" 7 0.17 5.0
run_one "kfb121" "contiguous" 8 0.18 2.0
run_one "kfb122" "contiguous" 8 0.18 3.0
run_one "kfb123" "contiguous" 8 0.18 4.0
run_one "kfb124" "contiguous" 8 0.18 5.0
run_one "kfb125" "contiguous" 8 0.20 2.0
run_one "kfb126" "contiguous" 8 0.20 3.0
run_one "kfb127" "contiguous" 8 0.20 4.0
run_one "kfb128" "contiguous" 8 0.20 5.0
run_one "kfb129" "contiguous" 8 0.22 2.0
run_one "kfb130" "contiguous" 8 0.22 3.0
run_one "kfb131" "contiguous" 8 0.22 4.0
run_one "kfb132" "contiguous" 8 0.22 5.0
run_one "kfb133" "contiguous" 10 0.28 2.0
run_one "kfb134" "contiguous" 10 0.28 3.0
run_one "kfb135" "contiguous" 10 0.28 4.0
run_one "kfb136" "contiguous" 10 0.28 5.0
run_one "kfb137" "contiguous" 10 0.31 2.0
run_one "kfb138" "contiguous" 10 0.31 3.0
run_one "kfb139" "contiguous" 10 0.31 4.0
run_one "kfb140" "contiguous" 10 0.31 5.0
run_one "kfb141" "contiguous" 10 0.34 2.0
run_one "kfb142" "contiguous" 10 0.34 3.0
run_one "kfb143" "contiguous" 10 0.34 4.0
run_one "kfb144" "contiguous" 10 0.34 5.0

echo "All runs completed. Building table..."
# write header
cat > "$MD_PATH" << 'EOF'
| Code | Placement | n | a_frac | sigma | Val Acc@1 | Val Acc@5 |
|------|-----------|---|--------|-------|-----------|-----------|
EOF

# append each row with best metrics parsed from logs
for i in "${!codes[@]}"; do
  code="${codes[$i]}"; placement="${placements[$i]}"; n="${ns[$i]}"; a="${afs[$i]}"; sigma="${sigmas[$i]}"
  read -r best_t1 best_t5 < <(get_best_metrics "$code.log")
  printf '| %s | %s | %s | %.2f | %s | %.3f | %.3f |
' "$code" "$placement" "$n" "$a" "$sigma" "$best_t1" "$best_t5" >> "$MD_PATH"
done

echo "Wrote table to $MD_PATH"
