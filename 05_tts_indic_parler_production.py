#!/usr/bin/env python3
"""
05_tts_indic_parler_production.py
Production natural-audio TTS runner for AI4Bharat Indic Parler-TTS.

Design goal:
- Reads translated segment JSON from Phase 04.
- Generates natural Indian-language speech using ai4bharat/indic-parler-tts.
- Does NOT time-scale, stretch, trim, or force audio into source timestamps by default.
- Concatenates generated segment audio with a small natural pause.
- Exports WAV, optional MP3, and a manifest JSON.

Example:
python 05_tts_indic_parler_production.py \
  --input-json /mnt/d/aistudio/audio_pipeline/output/translations/demo_hindi_segments.json \
  --output-dir /mnt/d/aistudio/audio_pipeline/output/tts \
  --output-name demo_hindi_indic_parler \
  --description "A female Indian speaker named Divya speaks in a warm narration style, at a moderate pace, with very clear audio, close microphone recording, and almost no background noise." \
  --export-sample-rate 48000 \
  --audio-channels 1 \
  --natural-pause-seconds 0.25 \
  --export-mp3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration


TEXT_KEYS = [
    "translated_text",
    "translation",
    "target_text",
    "text",
    "sentence",
    "transcript",
]

DEFAULT_DESCRIPTION = (
    "A female Indian speaker speaks in a clear narration style, at a moderate pace, "
    "with balanced pitch, very clear audio, close microphone recording, and almost no background noise."
)

LANGUAGE_DESCRIPTION_HINTS = {
    "hindi": "The speaker pronounces Hindi words naturally and clearly.",
    "kannada": "The speaker pronounces Kannada words naturally and clearly.",
    "telugu": "The speaker pronounces Telugu words naturally and clearly.",
    "tamil": "The speaker pronounces Tamil words naturally and clearly.",
    "malayalam": "The speaker pronounces Malayalam words naturally and clearly.",
    "marathi": "The speaker pronounces Marathi words naturally and clearly.",
    "bengali": "The speaker pronounces Bengali words naturally and clearly.",
    "gujarati": "The speaker pronounces Gujarati words naturally and clearly.",
    "odia": "The speaker pronounces Odia words naturally and clearly.",
    "urdu": "The speaker pronounces Urdu words naturally and clearly.",
    "english": "The speaker uses a natural Indian English accent.",
}


@dataclass
class SegmentResult:
    index: int
    text: str
    chunk_count: int
    sample_rate: int
    duration_seconds: float
    wav_path: str
    ok: bool
    error: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate natural translated audio using AI4Bharat Indic Parler-TTS."
    )
    parser.add_argument("--input-json", required=True, help="Phase 04 translated segments JSON path.")
    parser.add_argument("--output-dir", required=True, help="Folder where TTS output will be written.")
    parser.add_argument("--output-name", required=True, help="Base name for final output files.")
    parser.add_argument("--model", default="ai4bharat/indic-parler-tts", help="HF model id.")
    parser.add_argument("--language", default="", help="Optional language label for metadata and description hint.")
    parser.add_argument("--description", default=DEFAULT_DESCRIPTION, help="Voice/style caption for Parler-TTS.")
    parser.add_argument("--description-file", default="", help="Optional text file containing the voice/style caption.")
    parser.add_argument("--speaker-name", default="", help="Optional speaker name to inject into description, e.g. Divya, Maya, Rohit.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Inference device.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"], help="Torch dtype.")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Max generated audio tokens per chunk.")
    parser.add_argument("--max-chars-per-chunk", type=int, default=220, help="Split long text into chunks.")
    parser.add_argument("--natural-pause-seconds", type=float, default=0.25, help="Pause inserted between segments.")
    parser.add_argument("--chunk-pause-seconds", type=float, default=0.08, help="Pause inserted between chunks inside one segment.")
    parser.add_argument("--export-sample-rate", type=int, default=48000, help="Final WAV sample rate using ffmpeg resampling.")
    parser.add_argument("--audio-channels", type=int, default=1, choices=[1, 2], help="Final output channels.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for repeatability.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Generation temperature.")
    parser.add_argument("--do-sample", action="store_true", help="Use sampling. Default uses model defaults without forcing sampling.")
    parser.add_argument("--export-mp3", action="store_true", help="Also export MP3.")
    parser.add_argument("--keep-segment-wavs", action="store_true", help="Keep individual segment WAV files.")
    parser.add_argument("--skip-empty", action="store_true", default=True, help="Skip empty translated segments.")
    parser.add_argument("--limit", type=int, default=0, help="Debug: process only first N segments.")
    return parser.parse_args()


def read_text_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def unwrap_segments(data: Any) -> List[Dict[str, Any]]:
    """Accept common Phase 03/04 shapes: list, {segments:[...]}, {data:{segments:[...]}}."""
    if isinstance(data, list):
        return [x if isinstance(x, dict) else {"text": str(x)} for x in data]
    if isinstance(data, dict):
        for key in ["segments", "translated_segments", "items", "data"]:
            if key in data:
                value = data[key]
                if key == "data" and isinstance(value, dict):
                    nested = unwrap_segments(value)
                    if nested:
                        return nested
                if isinstance(value, list):
                    return [x if isinstance(x, dict) else {"text": str(x)} for x in value]
    raise ValueError("Could not find segments in JSON. Expected a list or a dict with key 'segments'.")


def pick_text(segment: Dict[str, Any]) -> str:
    for key in TEXT_KEYS:
        val = segment.get(key)
        if isinstance(val, str) and val.strip():
            return clean_text(val)
    # Some scripts store nested translated object
    for val in segment.values():
        if isinstance(val, dict):
            nested = pick_text(val)
            if nested:
                return nested
    return ""


def clean_text(text: str) -> str:
    text = text.replace("\u200c", "").replace("\u200d", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_text(text: str, max_chars: int) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    # Prefer sentence/punctuation boundaries, then comma, then spaces.
    parts = re.split(r"(?<=[।.!?])\s+", text)
    chunks: List[str] = []
    current = ""

    def flush_current() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
            current = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) > max_chars:
            flush_current()
            subparts = re.split(r"(?<=[,;:])\s+", part)
            for sub in subparts:
                sub = sub.strip()
                if len(sub) <= max_chars:
                    if len(current) + len(sub) + 1 <= max_chars:
                        current = (current + " " + sub).strip()
                    else:
                        flush_current()
                        current = sub
                else:
                    flush_current()
                    words = sub.split()
                    buf = ""
                    for word in words:
                        if len(buf) + len(word) + 1 <= max_chars:
                            buf = (buf + " " + word).strip()
                        else:
                            if buf:
                                chunks.append(buf)
                            buf = word
                    if buf:
                        chunks.append(buf)
        else:
            if len(current) + len(part) + 1 <= max_chars:
                current = (current + " " + part).strip()
            else:
                flush_current()
                current = part
    flush_current()
    return chunks


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda:0")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_dtype(dtype_arg: str, device: torch.device):
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if dtype_arg == "float32":
        return torch.float32
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def silence(seconds: float, sr: int) -> np.ndarray:
    if seconds <= 0:
        return np.zeros(0, dtype=np.float32)
    return np.zeros(int(seconds * sr), dtype=np.float32)


def normalize_audio_array(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0).astype(np.float32)
    # Safety only: prevent clipping, not loudness scaling.
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return audio


def run_ffmpeg_convert(input_wav: Path, output_wav: Path, sample_rate: int, channels: int) -> None:
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(input_wav),
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-sample_fmt", "s16",
        str(output_wav),
    ]
    subprocess.run(cmd, check=True)


def run_ffmpeg_mp3(input_wav: Path, output_mp3: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(input_wav),
        "-codec:a", "libmp3lame",
        "-q:a", "2",
        str(output_mp3),
    ]
    subprocess.run(cmd, check=True)


def build_description(args: argparse.Namespace) -> str:
    if args.description_file:
        description = read_text_file(args.description_file)
    else:
        description = args.description.strip() or DEFAULT_DESCRIPTION

    if args.speaker_name and args.speaker_name.lower() not in description.lower():
        description = f"{args.speaker_name}'s voice is used. {description}"

    lang = args.language.strip().lower()
    if lang in LANGUAGE_DESCRIPTION_HINTS and LANGUAGE_DESCRIPTION_HINTS[lang] not in description:
        description = f"{description} {LANGUAGE_DESCRIPTION_HINTS[lang]}"

    if "clear audio" not in description.lower():
        description = f"{description} The recording has very clear audio."

    return clean_text(description)


class IndicParlerEngine:
    def __init__(self, model_id: str, device: torch.device, dtype: Any):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        print(f"Loading model: {model_id}")
        print(f"Device: {device}; dtype: {dtype}")
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.description_tokenizer = AutoTokenizer.from_pretrained(self.model.config.text_encoder._name_or_path)
        self.sample_rate = int(self.model.config.sampling_rate)
        self.model.eval()

    @torch.inference_mode()
    def synthesize(self, text: str, description: str, max_new_tokens: int, temperature: float, do_sample: bool) -> np.ndarray:
        desc = self.description_tokenizer(description, return_tensors="pt")
        prompt = self.tokenizer(text, return_tensors="pt")
        desc = {k: v.to(self.device) for k, v in desc.items()}
        prompt = {k: v.to(self.device) for k, v in prompt.items()}

        kwargs = {
            "input_ids": desc["input_ids"],
            "attention_mask": desc.get("attention_mask"),
            "prompt_input_ids": prompt["input_ids"],
            "prompt_attention_mask": prompt.get("attention_mask"),
            "max_new_tokens": max_new_tokens,
        }
        if do_sample:
            kwargs["do_sample"] = True
            kwargs["temperature"] = temperature

        generation = self.model.generate(**kwargs)
        audio = generation.detach().cpu().numpy().squeeze()
        return normalize_audio_array(audio)


def main() -> int:
    args = parse_args()
    input_json = Path(args.input_json)
    output_dir = Path(args.output_dir)
    temp_dir = output_dir / f"{args.output_name}_segments"
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg before running this script.")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data = load_json(input_json)
    segments = unwrap_segments(data)
    if args.limit and args.limit > 0:
        segments = segments[: args.limit]

    texts = [pick_text(seg) for seg in segments]
    if args.skip_empty:
        indexed_texts = [(i, t) for i, t in enumerate(texts) if t.strip()]
    else:
        indexed_texts = [(i, t) for i, t in enumerate(texts)]

    if not indexed_texts:
        raise ValueError("No non-empty text found in input JSON.")

    description = build_description(args)
    device = get_device(args.device)
    dtype = get_dtype(args.dtype, device)
    engine = IndicParlerEngine(args.model, device, dtype)

    segment_results: List[SegmentResult] = []
    final_parts: List[np.ndarray] = []
    sr = engine.sample_rate

    print("\nVoice description:")
    print(description)
    print(f"\nSegments to synthesize: {len(indexed_texts)}")
    print("Natural mode: no scaling, no stretching, no source timestamp placement.\n")

    for idx, text in tqdm(indexed_texts, desc="Synthesizing segments"):
        segment_wav = temp_dir / f"segment_{idx:05d}.wav"
        try:
            chunks = split_text(text, args.max_chars_per_chunk)
            chunk_audios: List[np.ndarray] = []
            for chunk_num, chunk_text in enumerate(chunks, start=1):
                audio = engine.synthesize(
                    chunk_text,
                    description=description,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    do_sample=args.do_sample,
                )
                chunk_audios.append(audio)
                if chunk_num < len(chunks):
                    chunk_audios.append(silence(args.chunk_pause_seconds, sr))

            seg_audio = np.concatenate(chunk_audios) if chunk_audios else silence(args.natural_pause_seconds, sr)
            sf.write(segment_wav, seg_audio, sr)
            final_parts.append(seg_audio)
            final_parts.append(silence(args.natural_pause_seconds, sr))
            duration = float(len(seg_audio) / sr) if sr else 0.0
            segment_results.append(
                SegmentResult(
                    index=idx,
                    text=text,
                    chunk_count=len(chunks),
                    sample_rate=sr,
                    duration_seconds=duration,
                    wav_path=str(segment_wav),
                    ok=True,
                )
            )
        except Exception as e:
            segment_results.append(
                SegmentResult(
                    index=idx,
                    text=text,
                    chunk_count=0,
                    sample_rate=sr,
                    duration_seconds=0.0,
                    wav_path=str(segment_wav),
                    ok=False,
                    error=repr(e),
                )
            )
            print(f"\nERROR on segment {idx}: {e}", file=sys.stderr)

    if not final_parts:
        raise RuntimeError("No audio was generated.")

    final_audio = np.concatenate(final_parts).astype(np.float32)
    raw_wav = output_dir / f"{args.output_name}_raw_{sr}hz.wav"
    final_wav = output_dir / f"{args.output_name}.wav"
    sf.write(raw_wav, final_audio, sr)

    run_ffmpeg_convert(raw_wav, final_wav, args.export_sample_rate, args.audio_channels)

    mp3_path = None
    if args.export_mp3:
        mp3_path = output_dir / f"{args.output_name}.mp3"
        run_ffmpeg_mp3(final_wav, mp3_path)

    manifest = {
        "created_at_epoch": time.time(),
        "input_json": str(input_json),
        "model": args.model,
        "language": args.language,
        "description": description,
        "natural_mode_no_scaling": True,
        "model_sample_rate": sr,
        "export_sample_rate": args.export_sample_rate,
        "audio_channels": args.audio_channels,
        "final_wav": str(final_wav),
        "final_mp3": str(mp3_path) if mp3_path else None,
        "segment_count_input": len(segments),
        "segment_count_synthesized": len(indexed_texts),
        "success_count": sum(1 for r in segment_results if r.ok),
        "failed_count": sum(1 for r in segment_results if not r.ok),
        "segments": [asdict(r) for r in segment_results],
    }
    manifest_path = output_dir / f"{args.output_name}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.keep_segment_wavs:
        # Keep manifest paths truthful only if user asked to keep them. Otherwise remove temporary segment WAVs.
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("\nDone.")
    print(f"WAV: {final_wav}")
    if mp3_path:
        print(f"MP3: {mp3_path}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
