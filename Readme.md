# Phase 5B — Indic Parler-TTS Local Production Setup

This phase adds AI4Bharat Indic Parler-TTS to the existing local dubbing pipeline.

## Files

- `00_wsl2_indic_parler_tts_setup.sh` — one-time WSL2 setup script.
- `05_tts_indic_parler_production.py` — production TTS runner.

## Install

Copy both files to your WSL2 pipeline scripts folder:

```bash
cp 00_wsl2_indic_parler_tts_setup.sh /mnt/d/aistudio/audio_pipeline/scripts/
cp 05_tts_indic_parler_production.py /mnt/d/aistudio/audio_pipeline/scripts/
chmod +x /mnt/d/aistudio/audio_pipeline/scripts/00_wsl2_indic_parler_tts_setup.sh
chmod +x /mnt/d/aistudio/audio_pipeline/scripts/05_tts_indic_parler_production.py
```

Run setup once:

```bash
bash /mnt/d/aistudio/audio_pipeline/scripts/00_wsl2_indic_parler_tts_setup.sh
```

Accept the model access conditions on Hugging Face, then login:

```bash
source /mnt/d/aistudio/audio_pipeline/scripts/activate_indic_parler_tts.sh
hf auth login
```

## Run Hindi natural narration

```bash
source /mnt/d/aistudio/audio_pipeline/scripts/activate_indic_parler_tts.sh

python /mnt/d/aistudio/audio_pipeline/scripts/05_tts_indic_parler_production.py \
  --language hindi \
  --input-json /mnt/d/aistudio/audio_pipeline/output/translations/demo_hindi_segments.json \
  --output-dir /mnt/d/aistudio/audio_pipeline/output/tts \
  --output-name demo_hindi_indic_parler \
  --description "Divya's voice is warm and clear. She speaks Hindi in a natural narration style, at a moderate pace, with very clear audio, close microphone recording, and almost no background noise." \
  --export-sample-rate 48000 \
  --audio-channels 1 \
  --natural-pause-seconds 0.25 \
  --export-mp3
```

## Run Kannada natural narration

```bash
source /mnt/d/aistudio/audio_pipeline/scripts/activate_indic_parler_tts.sh

python /mnt/d/aistudio/audio_pipeline/scripts/05_tts_indic_parler_production.py \
  --language kannada \
  --input-json /mnt/d/aistudio/audio_pipeline/output/translations/demo_kannada_segments.json \
  --output-dir /mnt/d/aistudio/audio_pipeline/output/tts \
  --output-name demo_kannada_indic_parler \
  --description "A female Indian speaker speaks Kannada in a calm educational narration style, at a moderate pace, with very clear audio, close microphone recording, and almost no background noise." \
  --export-sample-rate 48000 \
  --audio-channels 1 \
  --natural-pause-seconds 0.25 \
  --export-mp3
```

## Notes

This script deliberately avoids duration scaling. It does not stretch, compress, trim, or force generated audio into original segment timings. It creates clean natural translated narration audio.
