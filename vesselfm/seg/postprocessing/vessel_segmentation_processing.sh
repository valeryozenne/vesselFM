#!/usr/bin/env bash
#
# run_vessel_segmentation_pipeline.sh
#
# Runs the vessel segmentation post-processing pipeline on one or more
# "*_cleaned.nii.gz" masks. INPUT_PATH can be:
#   - a single mask file (e.g. .../028__..._cleaned.nii.gz)
#   - a directory containing mask file(s) directly inside it
#     (e.g. out/MWA20220203a/Pre) -- only that folder is scanned
#   - a directory scanned recursively with --recursive
#     (e.g. out, to batch-process every patient/series under it)
#
# mirroring a layout such as:
#
#   out/<Patient>/<Pre|Post>/<series>_cleaned.nii.gz
#
# For each mask found, runs in sequence:
#   1. resample_mask.py        input_cleaned.nii.gz        -> input_cleaned_sr.nii.gz
#   2. binary_opening.py       input_cleaned_sr.nii.gz     -> input_cleaned_sr_bo.nii.gz
#   3. smooth_segmentation.py  input_cleaned_sr_bo.nii.gz  -> input_cleaned_smooth.nii.gz
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
RESOLUTION="0.2"
JOBS=1
FORCE=0
CLEANUP=0
DRY_RUN=0
RECURSIVE=0
PYTHON_BIN="python"
SCRIPT_DIR="/workspace_QMRI/PROJECTS_DATA/2025_RECH_NA_CHAUPATAT/vesselFM_test_db"
SCRIPTS_DIR="$SCRIPT_DIR"
INPUT_PATH=""

LOG_DIR="logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE=""
STATUS_FILE=""

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
usage() {
cat <<EOF
Usage: $(basename "$0") [OPTIONS] INPUT_PATH

INPUT_PATH is either:
  - a single "*_cleaned.nii.gz" mask file, or
  - a directory: every "*_cleaned.nii.gz" file directly inside it is
    processed (add --recursive to also descend into subfolders).

Runs the vessel segmentation post-processing pipeline
(resample -> binary opening -> smoothing) on the matching mask(s).

Options:
  -r, --resolution FLOAT   Target resolution for resample_mask.py (default: 0.2)
  -R, --recursive          When INPUT_PATH is a directory, search it
                            recursively instead of just its top level
  -j, --jobs N             Number of masks processed in parallel (default: 1)
  -f, --force              Reprocess even if output files already exist
  -c, --cleanup            Remove *_sr.nii.gz / *_sr_bo.nii.gz once the
                            final smoothed mask is produced successfully
  -n, --dry-run            Print the commands that would run, without
                            executing them
  -p, --python BIN         Python interpreter to use (default: python3)
  -s, --scripts-dir DIR    Directory containing resample_mask.py,
                            binary_opening.py, smooth_segmentation.py
                            (default: directory containing this script)
  -h, --help               Show this help message

Examples:
  # Process every mask directly inside a single patient/timepoint folder
  $(basename "$0") out/MWA20220203a/Pre

  # Process a single specific mask file
  $(basename "$0") out/MWA20220203a/Pre/028__..._cleaned.nii.gz

  # Batch-process an entire "out" tree, 4 masks at a time
  $(basename "$0") --recursive --jobs 4 out
EOF
}

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
log() {
    local msg="[$(date +'%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    if [[ -n "${LOG_FILE:-}" ]]; then
        echo "$msg" >> "$LOG_FILE"
    fi
}

die() {
    log "ERROR: $*"
    exit 1
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--resolution)  RESOLUTION="$2"; shift 2 ;;
        -R|--recursive)   RECURSIVE=1; shift ;;
        -j|--jobs)        JOBS="$2"; shift 2 ;;
        -f|--force)       FORCE=1; shift ;;
        -c|--cleanup)     CLEANUP=1; shift ;;
        -n|--dry-run)     DRY_RUN=1; shift ;;
        -p|--python)      PYTHON_BIN="$2"; shift 2 ;;
        -s|--scripts-dir) SCRIPTS_DIR="$2"; shift 2 ;;
        -h|--help)        usage; exit 0 ;;
        --) shift; break ;;
        -*) echo "Unknown option: $1" >&2; usage; exit 1 ;;
        *) INPUT_PATH="$1"; shift ;;
    esac
done

[[ -n "$INPUT_PATH" ]] || { echo "Missing INPUT_PATH" >&2; usage; exit 1; }
[[ -e "$INPUT_PATH" ]] || die "Input path not found: $INPUT_PATH"
[[ "$JOBS" =~ ^[0-9]+$ && "$JOBS" -ge 1 ]] || die "--jobs must be a positive integer"

RESAMPLE_SCRIPT="$SCRIPTS_DIR/resample_mask.py"
OPENING_SCRIPT="$SCRIPTS_DIR/binary_opening.py"
SMOOTH_SCRIPT="$SCRIPTS_DIR/smooth_segmentation.py"

for f in "$RESAMPLE_SCRIPT" "$OPENING_SCRIPT" "$SMOOTH_SCRIPT"; do
    [[ -f "$f" ]] || die "Required script not found: $f (use --scripts-dir to point at the folder containing it)"
done

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pipeline_${TIMESTAMP}.log"
STATUS_FILE="$LOG_DIR/status_${TIMESTAMP}.tsv"
: > "$STATUS_FILE"

log "Starting vessel segmentation pipeline"
log "Input path: $INPUT_PATH | Recursive: $RECURSIVE | Resolution: $RESOLUTION | Jobs: $JOBS | Force: $FORCE | Cleanup: $CLEANUP | Dry-run: $DRY_RUN"
log "Log file: $LOG_FILE"

# ---------------------------------------------------------------------------
# Per-file pipeline
# ---------------------------------------------------------------------------
process_one() {
    local input="$1"
    local base="${input%_cleaned.nii.gz}"
    local sr="${base}_cleaned_sr.nii.gz"
    local bo="${base}_cleaned_sr_bo.nii.gz"
    local smooth="${base}_cleaned_smooth.nii.gz"

    run() {
        if [[ "$DRY_RUN" -eq 1 ]]; then
            log "DRY-RUN: $*"
            return 0
        fi
        "$@" >> "$LOG_FILE" 2>&1
    }

    log "=== Processing: $input ==="

    # Step 1: resample / super-resolution
    if [[ "$FORCE" -eq 0 && -f "$sr" ]]; then
        log "SKIP resample (already exists): $sr"
    else
        log "Step 1/3 resample_mask.py -> $sr"
        if ! run "$PYTHON_BIN" "$RESAMPLE_SCRIPT" "$input" --resolution "$RESOLUTION" --output "$sr"; then
            log "FAILED resample: $input"
            printf '%s\t%s\t%s\n' "$input" "FAILED" "resample" >> "$STATUS_FILE"
            return 1
        fi
    fi

    # Step 2: binary opening
    if [[ "$FORCE" -eq 0 && -f "$bo" ]]; then
        log "SKIP binary opening (already exists): $bo"
    else
        log "Step 2/3 binary_opening.py -> $bo"
        if ! run "$PYTHON_BIN" "$OPENING_SCRIPT" "$sr" --output "$bo"; then
            log "FAILED binary opening: $input"
            printf '%s\t%s\t%s\n' "$input" "FAILED" "binary_opening" >> "$STATUS_FILE"
            return 1
        fi
    fi

    # Step 3: smoothing
    if [[ "$FORCE" -eq 0 && -f "$smooth" ]]; then
        log "SKIP smoothing (already exists): $smooth"
    else
        log "Step 3/3 smooth_segmentation.py -> $smooth"
        if ! run "$PYTHON_BIN" "$SMOOTH_SCRIPT" "$bo" --output "$smooth"; then
            log "FAILED smoothing: $input"
            printf '%s\t%s\t%s\n' "$input" "FAILED" "smooth_segmentation" >> "$STATUS_FILE"
            return 1
        fi
    fi

    if [[ "$CLEANUP" -eq 1 && "$DRY_RUN" -eq 0 ]]; then
        log "Cleanup: removing intermediate files for $input"
        rm -f "$sr" "$bo"
    fi

    log "DONE: $input -> $smooth"
    printf '%s\t%s\n' "$input" "OK" >> "$STATUS_FILE"
}

export -f process_one log
export DRY_RUN FORCE CLEANUP RESOLUTION PYTHON_BIN
export RESAMPLE_SCRIPT OPENING_SCRIPT SMOOTH_SCRIPT
export LOG_FILE STATUS_FILE

# ---------------------------------------------------------------------------
# Find masks and run (sequential or parallel)
# ---------------------------------------------------------------------------
MASKS=()

if [[ -f "$INPUT_PATH" ]]; then
    case "$INPUT_PATH" in
        *_cleaned.nii.gz) MASKS=("$INPUT_PATH") ;;
        *) die "Input file does not look like a *_cleaned.nii.gz mask: $INPUT_PATH" ;;
    esac
elif [[ -d "$INPUT_PATH" ]]; then
    if [[ "$RECURSIVE" -eq 1 ]]; then
        mapfile -d '' -t MASKS < <(find "$INPUT_PATH" -type f -name "*_cleaned.nii.gz" -print0 | sort -z)
    else
        mapfile -d '' -t MASKS < <(find "$INPUT_PATH" -maxdepth 1 -type f -name "*_cleaned.nii.gz" -print0 | sort -z)
    fi
else
    die "Input path is neither a file nor a directory: $INPUT_PATH"
fi

if [[ "${#MASKS[@]}" -eq 0 ]]; then
    log "No *_cleaned.nii.gz masks found at $INPUT_PATH (use --recursive to search subfolders)"
    exit 0
fi

log "Found ${#MASKS[@]} mask(s) to process"

printf '%s\0' "${MASKS[@]}" | xargs -0 -P "$JOBS" -I{} bash -c 'process_one "$@"' _ {} || true

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
TOTAL="${#MASKS[@]}"
OK_COUNT="$(grep -c -P '\tOK$' "$STATUS_FILE" || true)"
FAIL_COUNT="$(grep -c -P '\tFAILED' "$STATUS_FILE" || true)"

log "=== Summary: $OK_COUNT/$TOTAL succeeded, $FAIL_COUNT failed ==="
if [[ "$FAIL_COUNT" -gt 0 ]]; then
    log "Failed files (see $LOG_FILE for details):"
    grep -P '\tFAILED' "$STATUS_FILE" | while IFS=$'\t' read -r f status step; do
        log "  - $f (step: $step)"
    done
    exit 1
fi

exit 0