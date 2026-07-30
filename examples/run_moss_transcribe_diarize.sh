#!/usr/bin/env bash
# MOSS-Transcribe-Diarize end-to-end Python inference example.
#
# Usage:
#   export MODEL_ID=OpenMOSS-Team/MOSS-Transcribe-Diarize   # or local model path
#   export AUDIO_PATH=/path/to/audio.wav
#   export OUTPUT_DIR=/path/to/output/moss
#   bash examples/run_moss_transcribe_diarize.sh
#
# Install deps first:
#   pip install -e ".[moss]"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

MODEL_ID="${MODEL_ID:-OpenMOSS-Team/MOSS-Transcribe-Diarize}"
AUDIO_PATH="${AUDIO_PATH:-/path/to/audio.wav}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/moss}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-bf16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-false}"

EXTRA_ARGS=()
if [ "${LOCAL_FILES_ONLY}" = "true" ]; then
  EXTRA_ARGS+=(--local-files-only)
fi

python3 -m whosaidwhat.e2e.moss_transcribe_diarize.infer \
  --model "${MODEL_ID}" \
  --audio "${AUDIO_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --export-srt \
  "${EXTRA_ARGS[@]}"

# Example output files:
#   ${OUTPUT_DIR}/<stem>.txt              raw MOSS transcript
#   ${OUTPUT_DIR}/<stem>.json             run metadata
#   ${OUTPUT_DIR}/<stem>_segments.json    parsed speaker segments
#   ${OUTPUT_DIR}/<stem>.srt              subtitle export (with --export-srt)
