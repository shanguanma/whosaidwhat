"""Python inference CLI for MOSS-Transcribe-Diarize (HuggingFace backend only)."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from .inference_utils import (
    DEFAULT_PROMPT,
    build_transcription_messages,
    dtype_from_name,
    generate_transcription,
    resolve_device,
)
from .subtitle import export_json, export_srt, subtitle_segments_from_transcript
from .transcript_parser import parse_transcript


@dataclass(slots=True)
class TranscriptionResult:
    text: str
    prompt_len: int
    generated_tokens: int
    elapsed_sec: float
    model: str
    audio: str

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "prompt_len": self.prompt_len,
            "generated_tokens": self.generated_tokens,
            "elapsed_sec": self.elapsed_sec,
            "model": self.model,
            "audio": self.audio,
        }


class MossTranscriber:
    """Lazy HuggingFace model runner for MOSS-Transcribe-Diarize."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "auto",
        dtype: str = "bf16",
        local_files_only: bool = False,
    ):
        self.model_path = str(Path(model_path).expanduser())
        self.device_name = device
        self.dtype_name = dtype
        self.local_files_only = local_files_only
        self._model = None
        self._processor = None
        self._device: torch.device | None = None
        self._dtype: torch.dtype | None = None

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        prompt: str = DEFAULT_PROMPT,
        max_length: int = 131072,
        max_new_tokens: int = 2048,
        do_sample: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        token_callback: Callable[[int], None] | None = None,
    ) -> TranscriptionResult:
        self._ensure_loaded()
        started = time.time()
        result = generate_transcription(
            self._model,
            self._processor,
            build_transcription_messages(audio_path, prompt),
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            top_k=top_k if do_sample else None,
            device=self._device,
            dtype=self._dtype,
            token_callback=token_callback,
        )
        return TranscriptionResult(
            text=result["text"],
            prompt_len=int(result["prompt_len"]),
            generated_tokens=int(result["generated_tokens"]),
            elapsed_sec=time.time() - started,
            model=self.model_path,
            audio=str(Path(audio_path).expanduser()),
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        device = resolve_device(self.device_name)
        dtype = dtype_from_name(self.dtype_name)
        if device.type == "cpu":
            dtype = torch.float32
        load_kwargs = {"trust_remote_code": True}
        if self.local_files_only:
            load_kwargs["local_files_only"] = True
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype="auto",
            **load_kwargs,
        )
        processor = AutoProcessor.from_pretrained(
            self.model_path,
            fix_mistral_regex=True,
            **load_kwargs,
        )
        self._model = model.to(dtype=dtype).to(device).eval()
        self._processor = processor
        self._device = device
        self._dtype = dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MOSS-Transcribe-Diarize Python inference (HF Transformers)."
    )
    parser.add_argument(
        "--model",
        default="OpenMOSS-Team/MOSS-Transcribe-Diarize",
        help="HuggingFace model id or local path",
    )
    parser.add_argument("--audio", required=True, help="Input audio/video path")
    parser.add_argument(
        "--output-dir",
        default="./output/moss",
        help="Directory for transcript.txt / segments.json / subtitles.srt",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Transcription prompt")
    parser.add_argument("--device", default="auto", help="cuda:0, cpu, or auto")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-length", type=int, default=131072)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--export-srt", action="store_true", help="Also export subtitles.srt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_path = Path(args.audio).expanduser()
    if not audio_path.exists():
        raise SystemExit(f"Audio not found: {audio_path}")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    transcriber = MossTranscriber(
        args.model,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )

    def on_tokens(count: int) -> None:
        print(f"\rGenerated tokens: {count}", end="", flush=True)

    print(f"Transcribing: {audio_path}")
    result = transcriber.transcribe(
        audio_path,
        prompt=args.prompt,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        token_callback=on_tokens,
    )
    print()

    stem = audio_path.stem
    transcript_path = output_dir / f"{stem}.txt"
    meta_path = output_dir / f"{stem}.json"
    segments_path = output_dir / f"{stem}_segments.json"

    transcript_path.write_text(result.text + "\n", encoding="utf-8")
    meta_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    segments = [seg.__dict__ for seg in parse_transcript(result.text)]
    segments_path.write_text(
        json.dumps(segments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Transcript: {transcript_path}")
    print(f"Metadata:   {meta_path}")
    print(f"Segments:   {segments_path}")
    print(f"Elapsed:    {result.elapsed_sec:.2f}s, tokens: {result.generated_tokens}")

    for seg in parse_transcript(result.text):
        print(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.speaker}: {seg.text}")

    if args.export_srt:
        subtitle_segments = subtitle_segments_from_transcript(result.text)
        srt_path = output_dir / f"{stem}.srt"
        export_srt(subtitle_segments, srt_path)
        export_json(subtitle_segments, output_dir / f"{stem}_subtitle.json")
        print(f"SRT:        {srt_path}")


if __name__ == "__main__":
    main()
