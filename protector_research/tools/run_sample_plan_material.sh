#!/bin/bash
# Re-run the sample plan campaigns behind Fig. 2-1 / 2-2 / 2-3 with the contact-material protector model.
#   P3 (no protector, damping), P2d (6 mm SRM, damping), P2sid (6 mm silicone, damping),
#   P4 (no protector, LaFan), P5 (no protector, AMP), P6 (per-link SRM, AMPx)
# Uses the unchanged orchestrator scripts/run_sample_plan.sh; only contact_material.enabled is switched on
# for the duration of the run (and switched off afterwards). Old datasets are not touched: output goes to
# ~/data/mujoco_logs/nmi_dataset_material/sample_plan_<ts>.
# Requires the simulator built from workspace branch protector-sim2 (contact_material support).
set -uo pipefail
WS=/home/wang22/engineai/engineai_ros2_workspace
PM=$WS/src/simulation/mujoco/assets/config/pm_v2.yaml
TS=${SAMPLE_PLAN_TS:-$(date +%Y%m%d_%H%M%S)}
ROOT=${SAMPLE_PLAN_ROOT:-$HOME/data/mujoco_logs/nmi_dataset_material/sample_plan_$TS}
ONLY=${ONLY_RE:-'^(P3|P2d|P2sid|P4|P5|P6)_'}
# MATERIALS: materials file in assets/config (or an absolute path); written to contact_material.file for the run
# and set back to protector_materials.yaml afterwards (e.g. MATERIALS=protector_materials_wood.yaml).
MATERIALS=${MATERIALS:-protector_materials.yaml}
mkdir -p "$ROOT"

set_material() {  # $1 = true|false, only inside the contact_material block
  python3 - "$PM" "$1" <<'PY'
import re, sys
p, val = sys.argv[1], sys.argv[2]
s = open(p).read()
s2, n = re.subn(r"(^contact_material:\n(?:[ \t].*\n|#.*\n)*?[ \t]+enabled:[ \t]*)(true|false)", rf"\g<1>{val}", s, count=1, flags=re.M)
if n != 1:
    sys.exit("contact_material.enabled not found")
open(p, "w").write(s2)
PY
}

set_file() {  # $1 = file name, only inside the contact_material block
  python3 - "$PM" "$1" <<'PY'
import re, sys
p, val = sys.argv[1], sys.argv[2]
s = open(p).read()
s2, n = re.subn(r"(^contact_material:\n(?:[ \t].*\n|#.*\n)*?[ \t]+file:[ \t]*)(\S+)", lambda m: m.group(1) + val, s, count=1, flags=re.M)
if n != 1:
    sys.exit("contact_material.file not found")
open(p, "w").write(s2)
PY
}

trap 'set_material false; set_file protector_materials.yaml; echo "[material] contact_material disabled, file reset"' EXIT
git -C "$WS" rev-parse --abbrev-ref HEAD > "$ROOT/workspace_branch.txt"
git -C "$WS" rev-parse HEAD >> "$ROOT/workspace_branch.txt"
MAT_PATH=$MATERIALS; [[ $MAT_PATH == /* ]] || MAT_PATH=$WS/src/simulation/mujoco/assets/config/$MATERIALS
cp "$MAT_PATH" "$ROOT/"
set_file "$MATERIALS"
set_material true
grep -A2 "^contact_material:" "$PM" | tee "$ROOT/contact_material_block.txt"
cd "$WS"
SAMPLE_PLAN_ROOT="$ROOT" SAMPLE_PLAN_TS="$TS" bash scripts/run_sample_plan.sh --only "$ONLY" "$@"
