#!/usr/bin/env bash
# Stage 2: run the OFFICIAL ConceptGraphs pipeline on one Replica scene.
# Usage: bash 02_run_conceptgraphs.sh room0
# Do not tune their configs; the point is the comparison, not the score.
set -e
SCENE=${1:-room0}

# ---- the only two lines you may need to edit ------------------------------
CG_REPO="$(pwd)/concept-graphs"
REPLICA_ROOT="$(pwd)/../data/replica"        # contains <scene>/frame*.jpg etc.
# ---------------------------------------------------------------------------

eval "$(conda shell.bash hook)"; conda activate cgraphs
OUT="$(pwd)/cg_out/$SCENE"; mkdir -p "$OUT"

echo "== ConceptGraphs on $SCENE =="
echo "repo: $CG_REPO"
echo "data: $REPLICA_ROOT/$SCENE"
test -d "$REPLICA_ROOT/$SCENE" || { echo "FAIL: scene data missing"; exit 1; }

# Their documented Replica entry point (README 'Run' section). Their demo
# scene IS room0; the dataset config ships with the repo. If module paths
# have drifted, run  python -m pip show concept-graphs  and check their
# README — edit only the module name below.
cd "$CG_REPO"
SECONDS=0
python -m conceptgraph.slam.rerun_realtime_mapping \
    dataset_root="$REPLICA_ROOT" scene_id="$SCENE" \
    save_dir="$OUT" 2>&1 | tee "$OUT/run.log" || {
  echo "---------------------------------------------------------------"
  echo "Entry point drifted from this script's assumption."
  echo "Found these candidate mains in their repo:"
  grep -rl "__main__" conceptgraph --include="*.py" | head -20
  echo "Consult their README 'Usage' section, update the python -m line."
  exit 1
}
echo "wall-clock: ${SECONDS}s  (record this — it goes in the systems table)"
ls -lh "$OUT"
echo "STAGE OK (02_run_conceptgraphs $SCENE)"
