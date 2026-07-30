# whosaidwhat

Integrating ASR, speaker diarization, and forced alignment for the **who said what** task.

Two workflows are supported:

- **Modular pipeline** (`src/whosaidwhat/pipe/`): ASR + diarization + CTC alignment
- **End-to-end model** (`src/whosaidwhat/e2e/moss_transcribe_diarize/`): [MOSS-Transcribe-Diarize](https://github.com/shanguanma/MOSS-Transcribe-Diarize) single-pass transcription + diarization

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
pip install -e ".[nemo,eval]"      # modular pipeline
pip install -e ".[moss]"           # MOSS-Transcribe-Diarize e2e inference
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

## MOSS-Transcribe-Diarize (e2e)

Single-model inference for long-form multi-speaker audio. Output format:

```
[start_time][Sxx]transcribed speech[end_time]
```

Install MOSS dependencies, then run:

```bash
pip install -e ".[moss]"

export MODEL_ID=OpenMOSS-Team/MOSS-Transcribe-Diarize   # or local model path
export AUDIO_PATH=/path/to/audio.wav
export OUTPUT_DIR=/path/to/output/moss

bash examples/run_moss_transcribe_diarize.sh
```

Or invoke directly:

```bash
python -m whosaidwhat.e2e.moss_transcribe_diarize.infer \
  --model "$MODEL_ID" \
  --audio "$AUDIO_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --max-new-tokens 2048 \
  --export-srt
```

For long audio, increase `--max-new-tokens` (e.g. `65536`). Optional hotwords can be appended to `--prompt`.

Python API:

```python
from whosaidwhat.e2e.moss_transcribe_diarize import parse_transcript
from whosaidwhat.e2e.moss_transcribe_diarize.infer import MossTranscriber

transcriber = MossTranscriber("OpenMOSS-Team/MOSS-Transcribe-Diarize")
result = transcriber.transcribe("audio.wav", max_new_tokens=2048)
for seg in parse_transcript(result.text):
    print(seg.start, seg.end, seg.speaker, seg.text)
```

Requires Transformers 5.x and a CUDA GPU for best performance. vLLM / web app serving are not included in this repo.

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
src/whosaidwhat/e2e/moss_transcribe_diarize/  # MOSS-Transcribe-Diarize (HF inference only)
├── infer.py                       # CLI + MossTranscriber
├── inference_utils.py
├── transcript_parser.py
└── subtitle/                      # SRT/JSON export helpers
examples/
├── run_pipe.sh                    # modular pipeline benchmark workflow
└── run_moss_transcribe_diarize.sh # MOSS e2e inference example
```

## Credits

Derived from [whisper-diarization](https://github.com/MahmoudAshraf97/whisper-diarization), FireRedASR, NeMo Sortformer, MMS forced aligner, and [MOSS-Transcribe-Diarize](https://github.com/shanguanma/MOSS-Transcribe-Diarize).
