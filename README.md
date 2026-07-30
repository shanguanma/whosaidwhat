# whosaidwhat

Integrating ASR, speaker diarization, and forced alignment for the **who said what** task.

## Pipeline

```
Audio (WAV)
  → ASR (FireRedASR-LLM / FireRedASR2-LLM / Whisper)
  → Speaker diarization (NeMo Sortformer / MSDD / pyannote)
  → CTC forced alignment (MMS)
  → .txt / .srt
  → .stm (for meeteval evaluation)
```

## Install

```bash
pip install -e ".[nemo,eval]"
```

Core dependencies are in `pyproject.toml`. NeMo is required for diarization; `meeteval` is only needed for scoring.

The CTC aligner includes a pybind11 extension and is built automatically on install (requires a C++ compiler).

## Quick start

Set model and data paths, then run the example script:

```bash
export MODEL_DIR=/path/to/FireRedASR-LLM-L
export ALIGN_MODEL=/path/to/mms-300m-forced-aligner   # or HuggingFace model id
export INPUT_AUDIO_DIR=/path/to/dataset_root           # e.g. .../complete_audio
export OUTPUT_ROOT=/path/to/output
export SUBSET="mlc"                                    # dataset subset name

bash examples/run_pipe.sh --stage 300 --stop-stage 302
```

Without installing, add `src` to `PYTHONPATH` (the example script does this automatically).

### Single command (diarization + alignment)

```bash
python -m whosaidwhat.pipe.diarize_with_asr_improved \
  --model-dir-or-name "$MODEL_DIR" \
  --local-files-only true \
  --use-vad true \
  --diarizer streaming_softformer \
  --align-model "$ALIGN_MODEL" \
  --language en \
  --input-folder /path/to/wavs \
  --output-folder /path/to/output
```

Diarizer choices: `streaming_softformer`, `offline_softformer`, `msdd`, `pyannote31`.

### AliMeeting workflow (FireRedASR2 + FireRedVAD)

```bash
bash examples/run_pipe.sh --stage 330 --stop-stage 333
```

Uses `diarize_with_asr_improved2.py` (requires `--fireredvad-dir`) and `generate_hyp_stm_from_whisper_nemo_dia_output_for_alimeeting.py`.

Standalone VAD CLI (after `pip install -e .`):

```bash
fireredvad --task vad --wav_path /path/to.wav --model_dir /path/to/FireRedVAD/VAD
```

### SRT → STM

```bash
find /path/to/output -name "*.srt" > /path/to/output/srt_list.txt
python -m whosaidwhat.pipe.generate_hyp_stm_from_whisper_nemo_dia_output_ami_candor_fisher \
  /path/to/output/srt_list.txt /path/to/output/hyp.stm
```

## Project layout

```
src/whosaidwhat/pipe/
├── diarize_with_asr_improved.py   # main entry (FireRedASR-LLM)
├── diarize_with_asr_improved2.py  # FireRedASR2-LLM + FireRedVAD
├── helpers.py
├── generate_hyp_stm_*.py          # STM export for eval
├── fireredvad/                    # FireRedVAD (VAD/AED, used by diarize2)
├── ctc_forced_aligner/            # MMS CTC alignment
├── diarization/                   # NeMo MSDD diarizer
├── fireredasr/                    # FireRedASR v1 + VAD
└── fireredasr2/                   # FireRedASR2
examples/
└── run_pipe.sh                    # full benchmark-style workflow
```

## Credits

Derived from [whisper-diarization](https://github.com/MahmoudAshraf97/whisper-diarization), FireRedASR, NeMo Sortformer, and MMS forced aligner.
