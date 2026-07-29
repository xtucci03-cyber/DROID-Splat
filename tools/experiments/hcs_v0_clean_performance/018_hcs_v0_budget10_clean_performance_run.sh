#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT="/home/XT/gsslam/1_droidsplat"
readonly WT="$ROOT/code/DROID-Splat-hcs-v0-clean-perf-pair-v1"
readonly RUN_ID="018_perf_hcs_v0_budget10_no_activity_fr1desk_full_r1"
readonly REC="$ROOT/records/$RUN_ID"
readonly PY="/home/XT/.local/miniconda3-droidsplat/envs/droidsplat/bin/python"
readonly DATA="/home/XT/gsslam/3_datasets/TUM_RGBD-SLAM/rgbd_dataset_freiburg1_desk"
readonly ALGORITHM_PARENT_SHA="f3fb8d0dfb9bb1b774132777b7cfa3d7b2628230"
readonly EXPERIMENT_BRANCH="exp/hcs-v0-clean-performance-pair-v1"
readonly GPU_INDEX="0"
readonly HISTORY_BUDGET="10"
readonly RUN_ORDER_POSITION="FIRST"
readonly SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

test -d "$WT" || die "worktree does not exist: $WT"
test -d "$DATA" || die "dataset does not exist: $DATA"
test -x "$PY" || die "Python is not executable: $PY"
test ! -e "$REC" || die "record directory exists; refusing overwrite: $REC"

readonly ACTUAL_BRANCH="$(git -C "$WT" branch --show-current)"
readonly EXPERIMENT_PACKAGE_SHA="$(git -C "$WT" rev-parse HEAD)"
REMOTE_PACKAGE_SHA="$(
    git -C "$WT" rev-parse "refs/remotes/origin/$EXPERIMENT_BRANCH" 2>/dev/null
)" || die "missing fetched remote-tracking ref origin/$EXPERIMENT_BRANCH"
readonly REMOTE_PACKAGE_SHA

test "$ACTUAL_BRANCH" = "$EXPERIMENT_BRANCH" \
    || die "branch mismatch: $ACTUAL_BRANCH"
test "$EXPERIMENT_PACKAGE_SHA" = "$REMOTE_PACKAGE_SHA" \
    || die "HEAD does not match fetched origin branch"
test -z "$(git -C "$WT" status --short)" \
    || die "worktree is not clean"

mapfile -t commit_line < <(
    git -C "$WT" rev-list --parents -n 1 "$EXPERIMENT_PACKAGE_SHA"
)
read -r -a commit_parts <<< "${commit_line[0]}"
test "${#commit_parts[@]}" -eq 2 \
    || die "experiment package commit must have exactly one parent"
test "${commit_parts[1]}" = "$ALGORITHM_PARENT_SHA" \
    || die "algorithm parent mismatch: ${commit_parts[1]}"

readonly EXPECTED_PACKAGE_FILES="$(
    printf '%s\n' \
      "tools/experiments/hcs_v0_clean_performance/018_hcs_v0_budget10_clean_performance_run.sh" \
      "tools/experiments/hcs_v0_clean_performance/019_hcs_v0_budget20_clean_performance_run.sh" \
      "tools/experiments/hcs_v0_clean_performance/README.md" |
      LC_ALL=C sort
)"
readonly ACTUAL_PACKAGE_FILES="$(
    git -C "$WT" diff --name-only \
      "$ALGORITHM_PARENT_SHA..$EXPERIMENT_PACKAGE_SHA" |
      LC_ALL=C sort
)"
test "$ACTUAL_PACKAGE_FILES" = "$EXPECTED_PACKAGE_FILES" \
    || die "experiment package contains unexpected files"

git -C "$WT" diff --quiet \
    "$ALGORITHM_PARENT_SHA..$EXPERIMENT_PACKAGE_SHA" \
    -- configs src tools/analysis \
    || die "algorithm/config/parser files differ from algorithm parent"

mkdir -p "$REC"
exec > >(tee -a "$REC/launcher.log") 2>&1

MONITOR_PID=""
RUN_FINALIZED="false"

stop_monitor() {
    if test -n "$MONITOR_PID" && kill -0 "$MONITOR_PID" 2>/dev/null; then
        kill "$MONITOR_PID" 2>/dev/null || true
        wait "$MONITOR_PID" 2>/dev/null || true
    fi
    MONITOR_PID=""
}

preserve_failure() {
    local rc=$?
    local submodule_validation="false"
    stop_monitor
    if test -f "$REC/submodule_evidence_validation.json" &&
        grep -Eq '"validation_pass"[[:space:]]*:[[:space:]]*true' \
            "$REC/submodule_evidence_validation.json"; then
        submodule_validation="true"
    fi
    if test ! -f "$REC/run_end_utc.txt"; then
        date -u '+%Y-%m-%dT%H:%M:%SZ' > "$REC/run_end_utc.txt"
    fi
    if test "$RUN_FINALIZED" != "true" && test ! -f "$REC/RUN_STATUS.txt"; then
        {
            echo "RUN_ID=$RUN_ID"
            echo "EXPERIMENT_PACKAGE_SHA=$EXPERIMENT_PACKAGE_SHA"
            echo "ALGORITHM_PARENT_SHA=$ALGORITHM_PARENT_SHA"
            echo "HCS_MODE=deterministic_stratified"
            echo "HISTORY_BUDGET=$HISTORY_BUDGET"
            echo "SUBMODULE_EVIDENCE_METHOD=superproject_index_gitlinks_only"
            echo "RECURSIVE_SUBMODULE_STATUS_USED=false"
            echo "SUBMODULE_EVIDENCE_VALIDATION_PASS=$submodule_validation"
            echo "STATUS=FAIL_PRESERVED"
            echo "FAILURE_EXIT_CODE=$rc"
        } > "$REC/RUN_STATUS.txt"
    fi
    exit "$rc"
}
trap preserve_failure EXIT

collect_submodule_evidence() {
    local gitmodules_present="false"

    if git -C "$WT" cat-file -e HEAD:.gitmodules 2>/dev/null; then
        gitmodules_present="true"
        git -C "$WT" show HEAD:.gitmodules \
            > "$REC/gitmodules_snapshot.txt"
    else
        : > "$REC/gitmodules_snapshot.txt"
    fi

    git -C "$WT" ls-files --stage |
        awk '$1 == "160000" {
            print $1 "\t" $2 "\t" $4
        }' \
        > "$REC/submodule_gitlinks.txt"

    "$PY" - "$WT" "$REC" "$gitmodules_present" <<'SUBMODULE_PY'
from __future__ import annotations

import configparser
import json
import re
import sys
from collections import Counter
from pathlib import Path

worktree = Path(sys.argv[1])
record = Path(sys.argv[2])
gitmodules_present = sys.argv[3] == "true"
gitmodules_path = record / "gitmodules_snapshot.txt"
gitlinks_path = record / "submodule_gitlinks.txt"
errors: list[str] = []


def readable_text(path: Path, label: str) -> str:
    if not path.is_file():
        errors.append(f"{label}_MISSING")
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        errors.append(f"{label}_UNREADABLE={type(exc).__name__}")
        return ""


gitmodules_text = readable_text(
    gitmodules_path,
    "GITMODULES_SNAPSHOT",
)
gitlinks_text = readable_text(
    gitlinks_path,
    "SUBMODULE_GITLINKS",
)

gitmodules_paths: list[str] = []
if gitmodules_present:
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read_string(gitmodules_text)
        for section in parser.sections():
            if not section.startswith("submodule "):
                errors.append(f"INVALID_GITMODULES_SECTION={section}")
                continue
            if not parser.has_option(section, "path"):
                errors.append(f"GITMODULES_PATH_MISSING={section}")
                continue
            path = parser.get(section, "path").strip()
            if not path:
                errors.append(f"GITMODULES_PATH_EMPTY={section}")
                continue
            gitmodules_paths.append(path)
    except configparser.Error as exc:
        errors.append(f"GITMODULES_PARSE_ERROR={type(exc).__name__}")
elif gitmodules_text:
    errors.append("GITMODULES_CONTENT_WITHOUT_HEAD_ENTRY")

gitmodule_path_counts = Counter(gitmodules_paths)
duplicate_gitmodules_paths = sorted(
    path for path, count in gitmodule_path_counts.items() if count > 1
)
if duplicate_gitmodules_paths:
    errors.append(
        "DUPLICATE_GITMODULES_PATHS="
        + ",".join(duplicate_gitmodules_paths)
    )

gitlink_records: list[dict[str, str]] = []
malformed_gitlink_lines: list[int] = []
for line_number, raw_line in enumerate(
    gitlinks_text.splitlines(),
    start=1,
):
    fields = raw_line.split("\t", 2)
    if len(fields) != 3:
        malformed_gitlink_lines.append(line_number)
        continue
    mode, sha, path = fields
    if mode != "160000":
        errors.append(f"INVALID_GITLINK_MODE_LINE_{line_number}={mode}")
    if re.fullmatch(r"[0-9a-fA-F]{40}", sha) is None:
        errors.append(f"INVALID_GITLINK_SHA_LINE_{line_number}={sha}")
    if not path:
        errors.append(f"EMPTY_GITLINK_PATH_LINE_{line_number}")
    gitlink_records.append(
        {
            "mode": mode,
            "sha": sha.lower(),
            "path": path,
        }
    )
if malformed_gitlink_lines:
    errors.append(
        "MALFORMED_GITLINK_LINES="
        + ",".join(map(str, malformed_gitlink_lines))
    )

gitlink_paths = [record["path"] for record in gitlink_records]
gitlink_path_counts = Counter(gitlink_paths)
duplicate_paths = sorted(
    path for path, count in gitlink_path_counts.items() if count > 1
)
if duplicate_paths:
    errors.append("DUPLICATE_GITLINK_PATHS=" + ",".join(duplicate_paths))

gitlink_path_set = set(gitlink_paths)
missing_gitlinks = sorted(set(gitmodules_paths) - gitlink_path_set)
if missing_gitlinks:
    errors.append("MISSING_GITLINKS=" + ",".join(missing_gitlinks))

missing_worktree_directories = sorted(
    path
    for path in gitlink_paths
    if not (worktree / path).is_dir()
)
if missing_worktree_directories:
    errors.append(
        "MISSING_WORKTREE_DIRECTORIES="
        + ",".join(missing_worktree_directories)
    )

result = {
    "validation_pass": not errors,
    "collection_method": "superproject_index_gitlinks_only",
    "recursive_submodule_status_used": False,
    "gitmodules_present": gitmodules_present,
    "gitmodules_path_count": len(gitmodules_paths),
    "gitlink_count": len(gitlink_records),
    "gitmodules_paths": sorted(gitmodules_paths),
    "gitlink_paths": sorted(gitlink_paths),
    "gitlink_records": sorted(
        gitlink_records,
        key=lambda item: item["path"],
    ),
    "missing_gitlinks": missing_gitlinks,
    "duplicate_paths": duplicate_paths,
    "duplicate_gitmodules_paths": duplicate_gitmodules_paths,
    "missing_worktree_directories": missing_worktree_directories,
    "errors": errors,
}
(record / "submodule_evidence_validation.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(result, ensure_ascii=False, indent=2))
if errors:
    raise SystemExit(1)
print("SUBMODULE_EVIDENCE_VALIDATION_PASS")
SUBMODULE_PY
}

cp "$SCRIPT_PATH" "$REC/protocol_script.sh"
sha256sum "$SCRIPT_PATH" > "$REC/protocol_script.sha256"

{
    echo "RUN_ID=$RUN_ID"
    echo "EXPERIMENT_PACKAGE_SHA=$EXPERIMENT_PACKAGE_SHA"
    echo "ALGORITHM_PARENT_SHA=$ALGORITHM_PARENT_SHA"
    echo "EXPERIMENT_BRANCH=$EXPERIMENT_BRANCH"
    echo "WORKTREE=$WT"
    echo "DATASET=$DATA"
    echo "GPU_INDEX=$GPU_INDEX"
    echo "CUDA_VISIBLE_DEVICES=0"
    echo "PYTHONHASHSEED=43"
    echo "RUN_ORDER_POSITION=$RUN_ORDER_POSITION"
    echo "HCS_MODE=deterministic_stratified"
    echo "HISTORY_BUDGET=$HISTORY_BUDGET"
    echo "ACTIVITY_OBSERVER=false"
    echo "HCS_LOGGING=false"
    echo "LIFECYCLE_OBSERVER=false"
    echo "PERFORMANCE_MONITOR=true"
    echo "SUBMODULE_EVIDENCE_METHOD=superproject_index_gitlinks_only"
    echo "RECURSIVE_SUBMODULE_STATUS_USED=false"
    echo "MEASUREMENT_SCOPE=PAIRED_CLEAN_ALGORITHM_WITH_PERFORMANCE_MONITOR"
    echo "CLEAN_PAIR_ELIGIBLE=true"
    echo "PASS_SEALED_AUTOMATICALLY=false"
} > "$REC/code_state.txt"

git -C "$WT" status --short > "$REC/git_status_before.txt"
git -C "$WT" log -5 --decorate --oneline > "$REC/git_log_before.txt"
collect_submodule_evidence
git -C "$WT" remote -v > "$REC/git_remote.txt"
git -C "$WT" diff --name-status \
    "$ALGORITHM_PARENT_SHA..$EXPERIMENT_PACKAGE_SHA" \
    > "$REC/experiment_package_files.txt"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$REC/run_start_utc.txt"
"$PY" -VV > "$REC/python_version.txt" 2>&1
"$PY" -m pip freeze > "$REC/python_packages.txt" 2>&1 || true
nvidia-smi -q > "$REC/nvidia_smi_preflight.txt"

if ! nvidia-smi \
    --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits \
    -i "$GPU_INDEX" > "$REC/gpu_compute_apps_before.csv" 2>&1; then
    die "nvidia-smi compute-process preflight failed"
fi
if grep -Eq '^[[:space:]]*[0-9]+' "$REC/gpu_compute_apps_before.csv"; then
    die "GPU $GPU_INDEX already has a compute process"
fi

cat > "$REC/gpu_monitor.sh" <<'GPU_MONITOR'
#!/usr/bin/env bash
set -u

GPU_INDEX="$1"
ROOT_PID="$2"
GPU_OUT="$3"
PROCESS_OUT="$4"
CONTAMINATION_OUT="$5"

echo 'timestamp,index,utilization_gpu_percent,utilization_memory_percent,memory_used_mib,memory_total_mib,temperature_c,power_w' > "$GPU_OUT"
echo 'timestamp,pid,used_gpu_memory_mib,owned_by_run' > "$PROCESS_OUT"
echo 'timestamp,pid,used_gpu_memory_mib,reason' > "$CONTAMINATION_OUT"

is_descendant() {
    local current="$1"
    local parent=""
    while test "$current" -gt 1 2>/dev/null; do
        if test "$current" -eq "$ROOT_PID"; then
            return 0
        fi
        if test ! -r "/proc/$current/stat"; then
            return 1
        fi
        parent="$(awk '{print $4}' "/proc/$current/stat" 2>/dev/null)" || return 1
        current="$parent"
    done
    return 1
}

while true; do
    timestamp="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    nvidia-smi \
      --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
      --format=csv,noheader,nounits \
      -i "$GPU_INDEX" >> "$GPU_OUT" 2>/dev/null || true

    while IFS=',' read -r pid used; do
        pid="$(printf '%s' "$pid" | tr -d '[:space:]')"
        used="$(printf '%s' "$used" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        test "$pid" != "" || continue
        if is_descendant "$pid"; then
            echo "$timestamp,$pid,$used,1" >> "$PROCESS_OUT"
        else
            echo "$timestamp,$pid,$used,0" >> "$PROCESS_OUT"
            echo "$timestamp,$pid,$used,not_descendant_of_launcher" >> "$CONTAMINATION_OUT"
        fi
    done < <(
        nvidia-smi \
          --query-compute-apps=pid,used_gpu_memory \
          --format=csv,noheader,nounits \
          -i "$GPU_INDEX" 2>/dev/null || true
    )
    sleep 1
done
GPU_MONITOR
chmod +x "$REC/gpu_monitor.sh"

cat > "$REC/command.sh" <<COMMAND
#!/usr/bin/env bash
set -Eeuo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=43
export PYTHONPYCACHEPREFIX="$REC/pycache"

cd "$WT"
exec "$PY" run.py \\
  data=TUM_RGBD/fr1 \\
  data.input_folder="$DATA" \\
  tracking=tum \\
  mapping=tum \\
  mode=rgbd \\
  stride=1 \\
  run_frontend=True \\
  run_backend=True \\
  run_mapping=True \\
  run_loop_detection=False \\
  run_visualization=False \\
  run_mapping_gui=False \\
  show_stream=False \\
  backend_every=8 \\
  mapper_every=20 \\
  mapping.online_opt.iters=100 \\
  mapping.online_opt.n_last_frames=10 \\
  mapping.online_opt.n_rand_frames=20 \\
  mapping.online_opt.prune_every=2 \\
  mapping.online_opt.prune_densify_every=25 \\
  mapping.online_opt.prune_densify_until=55 \\
  mapping.online_opt.pruning.use_covisibility=True \\
  mapping.online_opt.pruning.covisibility.last=10 \\
  mapping.refinement.iters=0 \\
  mapping.refinement.sampling.use_non_keyframes=False \\
  evaluate=True \\
  render_images=True \\
  save_rendered_predictions=True \\
  +mapping.resource_admission.mode=observe \\
  +mapping.lifecycle_observer.enabled=false \\
  mapping.activity_observer.enabled=false \\
  mapping.camera_scheduler.enabled=true \\
  mapping.camera_scheduler.mode=deterministic_stratified \\
  mapping.camera_scheduler.history_budget=$HISTORY_BUDGET \\
  mapping.camera_scheduler.preserve_all_history_until=30 \\
  mapping.camera_scheduler.logging.enabled=false \\
  +mapping.performance_monitor.enabled=true \\
  +mapping.performance_monitor.gpu_timing=true \\
  +mapping.performance_monitor.memory_sampling=true \\
  +mapping.performance_monitor.log_events=true \\
  hydra.job.name="$RUN_ID" \\
  hydra.run.dir="$REC"
COMMAND
chmod +x "$REC/command.sh"

"$REC/gpu_monitor.sh" \
    "$GPU_INDEX" "$$" \
    "$REC/gpu_smi.csv" \
    "$REC/gpu_compute_processes.csv" \
    "$REC/gpu_contamination.csv" &
MONITOR_PID=$!

set +e
bash "$REC/command.sh" 2>&1 | tee "$REC/run.log"
DROID_EXIT="${PIPESTATUS[0]}"
set -e

stop_monitor
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$REC/run_end_utc.txt"
printf '%s\n' "$DROID_EXIT" > "$REC/droid_exit_code.txt"
git -C "$WT" status --short > "$REC/git_status_after.txt"

if test -f "$REC/.hydra/config.yaml"; then
    cp "$REC/.hydra/config.yaml" "$REC/hydra_config.yaml"
fi
if test -f "$REC/.hydra/overrides.yaml"; then
    cp "$REC/.hydra/overrides.yaml" "$REC/hydra_overrides.yaml"
fi

set +e
"$PY" "$WT/tools/analysis/build_performance_summary.py" \
    "$REC/run.log" \
    --output-jsonl "$REC/resource_monitor.jsonl" \
    --summary-json "$REC/performance_summary.json"
PARSER_EXIT=$?
set -e
printf '%s\n' "$PARSER_EXIT" > "$REC/performance_parser_exit_code.txt"

if test -d "$REC/evaluation"; then
    find "$REC/evaluation" -type f | wc -l | tr -d ' ' \
        > "$REC/evaluation_file_count.txt"
else
    echo 0 > "$REC/evaluation_file_count.txt"
fi
grep -Eic \
    'Traceback \(most recent call last\)|CUDA out of memory|Segmentation fault|Killed' \
    "$REC/run.log" > "$REC/fatal_pattern_count.txt" || true

"$PY" - "$REC" "$EXPERIMENT_PACKAGE_SHA" "$ALGORITHM_PARENT_SHA" \
    "$HISTORY_BUDGET" "$RUN_ORDER_POSITION" <<'PY'
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

from omegaconf import OmegaConf

rec = Path(sys.argv[1])
package_sha = sys.argv[2]
parent_sha = sys.argv[3]
history_budget = int(sys.argv[4])
run_order = sys.argv[5]
errors: list[str] = []


def read_int(name: str, default: int = -1) -> int:
    path = rec / name
    if not path.is_file():
        errors.append(f"MISSING={name}")
        return default
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        errors.append(f"INVALID_INT={name}")
        return default


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        errors.append(f"MISSING={path.relative_to(rec)}")
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str] | None, key: str):
    if row is None:
        return None
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        errors.append(f"INVALID_METRIC={key}")
        return None


def choose(rows: list[dict[str, str]], key: str, value: str):
    matches = [row for row in rows if row.get(key) == value]
    if len(matches) != 1:
        errors.append(f"EXPECTED_ONE_ROW={key}:{value}:got={len(matches)}")
        return None
    return matches[0]


droid_exit = read_int("droid_exit_code.txt")
parser_exit = read_int("performance_parser_exit_code.txt")
evaluation_file_count = read_int("evaluation_file_count.txt")
fatal_pattern_count = read_int("fatal_pattern_count.txt")
if droid_exit != 0:
    errors.append(f"DROID_EXIT={droid_exit}")
if parser_exit != 0:
    errors.append(f"PARSER_EXIT={parser_exit}")
if evaluation_file_count != 540:
    errors.append(f"EVALUATION_FILE_COUNT={evaluation_file_count}")
if fatal_pattern_count != 0:
    errors.append(f"FATAL_PATTERN_COUNT={fatal_pattern_count}")

required_files = [
    "run.log",
    "hydra_config.yaml",
    "hydra_overrides.yaml",
    "performance_summary.json",
    "resource_monitor.jsonl",
    "gitmodules_snapshot.txt",
    "submodule_gitlinks.txt",
    "submodule_evidence_validation.json",
    "evaluation/odometry/evaluation_results.csv",
    "evaluation/rendering/evaluation_results.csv",
]
for name in required_files:
    if not (rec / name).is_file():
        errors.append(f"MISSING={name}")

submodule_evidence = {}
submodule_evidence_path = rec / "submodule_evidence_validation.json"
if submodule_evidence_path.is_file():
    try:
        submodule_evidence = json.loads(
            submodule_evidence_path.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError) as exc:
        errors.append(f"SUBMODULE_EVIDENCE_LOAD_FAILED={type(exc).__name__}")
if not isinstance(submodule_evidence, dict):
    errors.append("SUBMODULE_EVIDENCE_NOT_OBJECT")
    submodule_evidence = {}

submodule_evidence_method = submodule_evidence.get("collection_method")
recursive_submodule_status_used = submodule_evidence.get(
    "recursive_submodule_status_used"
)
submodule_evidence_validation_pass = submodule_evidence.get("validation_pass")
if submodule_evidence_method != "superproject_index_gitlinks_only":
    errors.append(
        f"SUBMODULE_EVIDENCE_METHOD={submodule_evidence_method!r}"
    )
if recursive_submodule_status_used is not False:
    errors.append(
        "RECURSIVE_SUBMODULE_STATUS_USED="
        f"{recursive_submodule_status_used!r}"
    )
if submodule_evidence_validation_pass is not True:
    errors.append(
        "SUBMODULE_EVIDENCE_VALIDATION_PASS="
        f"{submodule_evidence_validation_pass!r}"
    )
if submodule_evidence.get("errors") != []:
    errors.append(
        f"SUBMODULE_EVIDENCE_ERRORS={submodule_evidence.get('errors')!r}"
    )

cfg = None
if (rec / "hydra_config.yaml").is_file():
    try:
        cfg = OmegaConf.load(rec / "hydra_config.yaml")
    except Exception as exc:
        errors.append(f"CONFIG_LOAD_FAILED={type(exc).__name__}")

if cfg is not None:
    checks = [
        (cfg.mode == "rgbd", "MODE_NOT_RGBD"),
        (int(cfg.stride) == 1, "STRIDE_NOT_1"),
        (cfg.run_frontend is True, "FRONTEND_NOT_TRUE"),
        (cfg.run_backend is True, "BACKEND_NOT_TRUE"),
        (cfg.run_mapping is True, "MAPPING_NOT_TRUE"),
        (cfg.run_loop_detection is False, "LOOP_DETECTION_NOT_FALSE"),
        (cfg.run_visualization is False, "VISUALIZATION_NOT_FALSE"),
        (cfg.run_mapping_gui is False, "MAPPING_GUI_NOT_FALSE"),
        (cfg.show_stream is False, "SHOW_STREAM_NOT_FALSE"),
        (int(cfg.backend_every) == 8, "BACKEND_EVERY_NOT_8"),
        (int(cfg.mapper_every) == 20, "MAPPER_EVERY_NOT_20"),
        (cfg.evaluate is True, "EVALUATE_NOT_TRUE"),
        (cfg.render_images is True, "RENDER_IMAGES_NOT_TRUE"),
        (
            cfg.save_rendered_predictions is True,
            "SAVE_RENDERED_PREDICTIONS_NOT_TRUE",
        ),
        (cfg.get("t_start", None) is None, "T_START_PRESENT"),
        (cfg.get("t_stop", None) is None, "T_STOP_PRESENT"),
        (int(cfg.mapping.online_opt.iters) == 100, "MAPPING_ITERS_NOT_100"),
        (
            int(cfg.mapping.online_opt.n_last_frames) == 10,
            "N_LAST_FRAMES_NOT_10",
        ),
        (
            int(cfg.mapping.online_opt.n_rand_frames) == 20,
            "N_RAND_FRAMES_NOT_20",
        ),
        (int(cfg.mapping.refinement.iters) == 0, "REFINEMENT_ITERS_NOT_0"),
        (
            cfg.mapping.refinement.sampling.use_non_keyframes is False,
            "REFINEMENT_NON_KEYFRAMES_NOT_FALSE",
        ),
        (
            cfg.mapping.online_opt.pruning.use_covisibility is True,
            "COVISIBILITY_NOT_TRUE",
        ),
        (cfg.mapping.resource_admission.mode == "observe", "M01_NOT_OBSERVE"),
        (
            cfg.mapping.lifecycle_observer.enabled is False,
            "LIFECYCLE_OBSERVER_NOT_FALSE",
        ),
        (
            cfg.mapping.activity_observer.enabled is False,
            "ACTIVITY_OBSERVER_NOT_FALSE",
        ),
        (
            cfg.mapping.camera_scheduler.enabled is True,
            "CAMERA_SCHEDULER_NOT_TRUE",
        ),
        (
            cfg.mapping.camera_scheduler.mode == "deterministic_stratified",
            "CAMERA_SCHEDULER_MODE_INVALID",
        ),
        (
            int(cfg.mapping.camera_scheduler.history_budget) == history_budget,
            "HISTORY_BUDGET_INVALID",
        ),
        (
            int(cfg.mapping.camera_scheduler.preserve_all_history_until) == 30,
            "PRESERVE_ALL_NOT_30",
        ),
        (
            cfg.mapping.camera_scheduler.logging.enabled is False,
            "HCS_LOGGING_NOT_FALSE",
        ),
    ]
    for passed, label in checks:
        if not passed:
            errors.append(label)
    perf_cfg = cfg.mapping.performance_monitor
    for key in ("enabled", "gpu_timing", "memory_sampling", "log_events"):
        if perf_cfg[key] is not True:
            errors.append(f"PERFORMANCE_MONITOR_{key.upper()}_NOT_TRUE")

log_text = (
    (rec / "run.log").read_text(encoding="utf-8", errors="replace")
    if (rec / "run.log").is_file()
    else ""
)
marker_counts = {
    "performance": log_text.count("[PerformanceResourceMonitor]"),
    "activity": log_text.count("[MappingActivityObserver]"),
    "hcs": log_text.count("[HistoricalCameraScheduler]"),
    "lifecycle": log_text.count("[LifecycleObserver]"),
}
if marker_counts["performance"] <= 0:
    errors.append("PERFORMANCE_MARKER_COUNT=0")
for marker in ("activity", "hcs", "lifecycle"):
    if marker_counts[marker] != 0:
        errors.append(f"UNEXPECTED_{marker.upper()}_MARKERS")

summary = {}
if (rec / "performance_summary.json").is_file():
    try:
        summary = json.loads(
            (rec / "performance_summary.json").read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError) as exc:
        errors.append(f"PERFORMANCE_SUMMARY_LOAD_FAILED={type(exc).__name__}")

validation = summary.get("validation", {})
if validation.get("validation_pass") is not True:
    errors.append("PERFORMANCE_VALIDATION_NOT_TRUE")
if validation.get("stage_pairs_complete") is not True:
    errors.append("STAGE_PAIRS_INCOMPLETE")
if validation.get("process_lifecycle_complete") is not True:
    errors.append("PROCESS_LIFECYCLE_INCOMPLETE")
if validation.get("process_count") != 4:
    errors.append(f"PROCESS_COUNT={validation.get('process_count')}")

mapper_update_ids = summary.get("mapper_update_ids")
if not isinstance(mapper_update_ids, list) or not mapper_update_ids:
    errors.append("MAPPER_UPDATE_IDS_EMPTY")
elif mapper_update_ids != list(range(len(mapper_update_ids))):
    errors.append(f"MAPPER_UPDATE_IDS_NOT_CONTIGUOUS={mapper_update_ids}")

stages = summary.get("stage_metrics", {})
required_stages = [
    "online_slam",
    "evaluation",
    "mapper_update",
    "mapping_optimization",
    "add_new_gaussians",
    "covisibility_pruning",
]
for stage in required_stages:
    if stage not in stages:
        errors.append(f"MISSING_STAGE={stage}")

monitor_events = []
if (rec / "resource_monitor.jsonl").is_file():
    try:
        monitor_events = [
            json.loads(line)
            for line in (rec / "resource_monitor.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
    except json.JSONDecodeError as exc:
        errors.append(f"RESOURCE_JSONL_INVALID={exc.lineno}")
online_end = [
    event
    for event in monitor_events
    if event.get("process_role") == "main"
    and event.get("stage") == "online_slam"
    and event.get("phase") == "end"
    and event.get("status") == "ok"
]
if len(online_end) != 1:
    errors.append(f"ONLINE_END_EVENT_COUNT={len(online_end)}")
elif online_end[0].get("frame_count") != 592:
    errors.append(f"FRAME_COUNT={online_end[0].get('frame_count')}")

contamination_rows = read_csv_rows(rec / "gpu_contamination.csv")
if contamination_rows:
    errors.append(f"GPU_CONTAMINATION_ROWS={len(contamination_rows)}")

gpu_rows = read_csv_rows(rec / "gpu_smi.csv")
gpu_memory = []
for row in gpu_rows:
    try:
        gpu_memory.append(float(row["memory_used_mib"]))
    except (KeyError, ValueError):
        errors.append("INVALID_GPU_SMI_MEMORY_ROW")
        break
if not gpu_memory:
    errors.append("GPU_SMI_SAMPLE_COUNT=0")

odom_rows = read_csv_rows(
    rec / "evaluation/odometry/evaluation_results.csv"
)
render_rows = read_csv_rows(
    rec / "evaluation/rendering/evaluation_results.csv"
)
key_odom = choose(odom_rows, "ate_on_keyframes_only", "True")
all_odom = choose(odom_rows, "ate_on_keyframes_only", "False")
key_render = choose(render_rows, "eval_on_keyframes", "True")
nonkey_render = choose(render_rows, "eval_on_keyframes", "False")

gaussian_matches = re.findall(
    r"Rendering finished with\s+([0-9]+)\s+Gaussians",
    log_text,
)
final_gaussians = int(gaussian_matches[-1]) if gaussian_matches else None
if final_gaussians is None:
    errors.append("FINAL_GAUSSIAN_NOT_FOUND")

git_after = (
    (rec / "git_status_after.txt").read_text(encoding="utf-8")
    if (rec / "git_status_after.txt").is_file()
    else "missing"
)
if git_after.strip():
    errors.append("WORKTREE_DIRTY_AFTER_RUN")


def stage_value(stage: str, field: str):
    value = stages.get(stage, {}).get(field)
    return value


key_metrics = {
    "online_slam_cpu_ms": stage_value("online_slam", "cpu_wall_total_ms"),
    "evaluation_cpu_ms": stage_value("evaluation", "cpu_wall_total_ms"),
    "mapper_update_cpu_ms": stage_value("mapper_update", "cpu_wall_total_ms"),
    "mapper_update_gpu_ms": stage_value("mapper_update", "gpu_elapsed_total_ms"),
    "mapping_optimization_cpu_ms": stage_value(
        "mapping_optimization", "cpu_wall_total_ms"
    ),
    "mapping_optimization_gpu_ms": stage_value(
        "mapping_optimization", "gpu_elapsed_total_ms"
    ),
    "add_new_gaussians_cpu_ms": stage_value(
        "add_new_gaussians", "cpu_wall_total_ms"
    ),
    "add_new_gaussians_gpu_ms": stage_value(
        "add_new_gaussians", "gpu_elapsed_total_ms"
    ),
    "covisibility_pruning_cpu_ms": stage_value(
        "covisibility_pruning", "cpu_wall_total_ms"
    ),
    "covisibility_pruning_gpu_ms": stage_value(
        "covisibility_pruning", "gpu_elapsed_total_ms"
    ),
    "mapper_max_memory_allocated_bytes": stage_value(
        "mapper_update", "max_memory_allocated_peak_bytes"
    ),
    "mapper_max_memory_reserved_bytes": stage_value(
        "mapper_update", "max_memory_reserved_peak_bytes"
    ),
    "nvidia_smi_sample_count": len(gpu_memory),
    "nvidia_smi_memory_peak_mib": max(gpu_memory) if gpu_memory else None,
    "nvidia_smi_memory_mean_mib": (
        sum(gpu_memory) / len(gpu_memory) if gpu_memory else None
    ),
    "final_gaussians": final_gaussians,
    "keyframe_ate": as_float(key_odom, "ate"),
    "all_frame_ate": as_float(all_odom, "ate"),
    "keyframe_psnr": as_float(key_render, "psnr"),
    "nonkeyframe_psnr": as_float(nonkey_render, "psnr"),
    "keyframe_ssim": as_float(key_render, "ssim"),
    "nonkeyframe_ssim": as_float(nonkey_render, "ssim"),
    "keyframe_lpips": as_float(key_render, "lpips"),
    "nonkeyframe_lpips": as_float(nonkey_render, "lpips"),
    "keyframe_l1_depth": as_float(key_render, "l1_depth"),
    "nonkeyframe_l1_depth": as_float(nonkey_render, "l1_depth"),
}

result = {
    "schema": 1,
    "run_id": rec.name,
    "experiment_package_sha": package_sha,
    "algorithm_parent_sha": parent_sha,
    "history_budget": history_budget,
    "run_order_position": run_order,
    "measurement_scope": "PAIRED_CLEAN_ALGORITHM_WITH_PERFORMANCE_MONITOR",
    "clean_pair_eligible": True,
    "performance_monitor_overhead_included": True,
    "droid_exit": droid_exit,
    "parser_exit": parser_exit,
    "evaluation_file_count": evaluation_file_count,
    "fatal_pattern_count": fatal_pattern_count,
    "marker_counts": marker_counts,
    "performance_event_count": summary.get("event_count"),
    "stage_pair_count": validation.get("stage_pair_count"),
    "process_count": validation.get("process_count"),
    "mapper_update_ids": mapper_update_ids,
    "submodule_evidence_method": submodule_evidence_method,
    "recursive_submodule_status_used": recursive_submodule_status_used,
    "submodule_evidence_validation_pass": submodule_evidence_validation_pass,
    "key_metrics": key_metrics,
    "errors": errors,
    "validation_pass": not errors,
}

(rec / "PERFORMANCE_VALIDATION.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
(rec / "PERFORMANCE_KEY_METRICS.json").write_text(
    json.dumps(key_metrics, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
status = "PASS_NOT_YET_SEALED" if not errors else "FAIL_PRESERVED"
(rec / "RUN_STATUS.txt").write_text(
    "\n".join(
        [
            f"RUN_ID={rec.name}",
            f"EXPERIMENT_PACKAGE_SHA={package_sha}",
            f"ALGORITHM_PARENT_SHA={parent_sha}",
            "HCS_MODE=deterministic_stratified",
            f"HISTORY_BUDGET={history_budget}",
            "ACTIVITY_OBSERVER=false",
            "HCS_LOGGING=false",
            "LIFECYCLE_OBSERVER=false",
            "PERFORMANCE_MONITOR=true",
            "PERFORMANCE_MONITOR_OVERHEAD_INCLUDED=true",
            "SUBMODULE_EVIDENCE_METHOD=superproject_index_gitlinks_only",
            "RECURSIVE_SUBMODULE_STATUS_USED=false",
            (
                "SUBMODULE_EVIDENCE_VALIDATION_PASS="
                f"{str(submodule_evidence_validation_pass is True).lower()}"
            ),
            f"DROID_EXIT={droid_exit}",
            f"PERFORMANCE_PARSER_EXIT={parser_exit}",
            f"EVALUATION_FILE_COUNT={evaluation_file_count}",
            f"VALIDATION_PASS={str(not errors).lower()}",
            f"STATUS={status}",
        ]
    )
    + "\n",
    encoding="utf-8",
)

summary_lines = [
    f"# {rec.name} 运行后摘要",
    "",
    f"- 状态：`{status}`",
    f"- 实验包提交：`{package_sha}`",
    f"- 算法父提交：`{parent_sha}`",
    f"- HCS history budget：`{history_budget}`",
    "- Activity/Lifecycle/HCS详细日志：关闭",
    "- Performance Resource Monitor：开启，开销包含在计时中",
    "- 子模块证据方法：`superproject_index_gitlinks_only`",
    "- 使用递归 `git submodule status`：`false`",
    (
        "- 子模块证据验证："
        f"`{str(submodule_evidence_validation_pass is True).lower()}`"
    ),
    f"- DROID退出码：`{droid_exit}`",
    f"- Parser退出码：`{parser_exit}`",
    f"- GPU污染事件：`{len(contamination_rows)}`",
    f"- Final Gaussians：`{final_gaussians}`",
    "",
    "## 关键指标",
    "",
    "```json",
    json.dumps(key_metrics, ensure_ascii=False, indent=2),
    "```",
    "",
    "## 验收错误",
    "",
]
summary_lines.extend(
    [f"- {error}" for error in errors]
    if errors
    else ["- 无；等待人工审核，尚未封存。"]
)
(rec / "POSTRUN_SUMMARY_中文.md").write_text(
    "\n".join(summary_lines) + "\n",
    encoding="utf-8",
)

print(json.dumps(result, ensure_ascii=False, indent=2))
if errors:
    raise SystemExit(1)
print("018_HCS_V0_BUDGET10_CLEAN_PERFORMANCE_VALIDATION_PASS")
PY

find "$REC" -type f \
    ! -name record_checksums.sha256 \
    ! -name launcher.log \
    -print0 |
    LC_ALL=C sort -z |
    xargs -0 sha256sum > "$REC/record_checksums.sha256"

RUN_FINALIZED="true"
trap - EXIT
cat "$REC/RUN_STATUS.txt"
echo "018_HCS_V0_BUDGET10_CLEAN_PERFORMANCE_RUN_COMPLETE"
