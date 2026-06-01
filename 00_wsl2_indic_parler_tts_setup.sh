#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 00_wsl2_indic_parler_tts_setup.sh
# One-time WSL2 setup for AI4Bharat Indic Parler-TTS production
# Target folder: /mnt/d/aistudio/audio_pipeline
# Conda env: indic-parler-tts
# ============================================================

ENV_NAME="${ENV_NAME:-indic-parler-tts}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
AISTUDIO_ROOT="${AISTUDIO_ROOT:-/mnt/d/aistudio}"
PIPELINE_ROOT="${PIPELINE_ROOT:-$AISTUDIO_ROOT/audio_pipeline}"
HF_HOME_DIR="${HF_HOME:-$PIPELINE_ROOT/models/huggingface}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

log() { echo -e "\n[Indic-Parler-Setup] $*"; }

log "Creating production folders..."
mkdir -p "$PIPELINE_ROOT"/{input/{audio,video,prompts},output/{transcripts,translations,tts,final_video},models/{huggingface,indic_parler_tts},scripts,logs,temp}
mkdir -p "$AISTUDIO_ROOT/conda_envs" "$AISTUDIO_ROOT/conda_pkgs"

log "Installing Ubuntu system packages..."
sudo apt-get update
sudo apt-get install -y \
  build-essential git git-lfs wget curl ffmpeg sox libsndfile1 espeak-ng \
  python3-dev pkg-config

git lfs install || true

log "Finding Conda..."
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  CONDA_BASE="$HOME/miniconda3"
elif [ -f "$HOME/mambaforge/etc/profile.d/conda.sh" ]; then
  CONDA_BASE="$HOME/mambaforge"
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  CONDA_BASE="$HOME/miniforge3"
else
  echo "ERROR: Conda was not found. Install Miniconda/Mambaforge in WSL2 first, then rerun this script."
  exit 1
fi

# shellcheck disable=SC1090
source "$CONDA_BASE/etc/profile.d/conda.sh"

log "Configuring Conda folders on D: drive..."
conda config --add envs_dirs "$AISTUDIO_ROOT/conda_envs" || true
conda config --add pkgs_dirs "$AISTUDIO_ROOT/conda_pkgs" || true

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  log "Conda environment already exists: $ENV_NAME"
else
  log "Creating Conda environment: $ENV_NAME with Python $PYTHON_VERSION"
  conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"

log "Upgrading pip tooling..."
python -m pip install --upgrade pip setuptools wheel

log "Installing PyTorch with CUDA wheels from: $TORCH_INDEX_URL"
python -m pip install --upgrade torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"

log "Installing Indic Parler-TTS and production audio dependencies..."
python -m pip install --upgrade \
  "git+https://github.com/huggingface/parler-tts.git" \
  transformers accelerate huggingface_hub safetensors sentencepiece protobuf \
  numpy scipy soundfile librosa pydub tqdm rich jsonschema unidecode

log "Writing environment helper file..."
cat > "$PIPELINE_ROOT/scripts/activate_indic_parler_tts.sh" <<EOF
#!/usr/bin/env bash
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
export AISTUDIO_ROOT="$AISTUDIO_ROOT"
export PIPELINE_ROOT="$PIPELINE_ROOT"
export HF_HOME="$HF_HOME_DIR"
export HF_HUB_CACHE="$HF_HOME_DIR/hub"
export TRANSFORMERS_CACHE="$HF_HOME_DIR"
export TOKENIZERS_PARALLELISM=false
EOF
chmod +x "$PIPELINE_ROOT/scripts/activate_indic_parler_tts.sh"

log "Persisting Hugging Face cache variables in ~/.bashrc if missing..."
grep -q "# AI Studio Indic Parler TTS" "$HOME/.bashrc" || cat >> "$HOME/.bashrc" <<EOF

# AI Studio Indic Parler TTS
export AISTUDIO_ROOT="$AISTUDIO_ROOT"
export PIPELINE_ROOT="$PIPELINE_ROOT"
export HF_HOME="$HF_HOME_DIR"
export HF_HUB_CACHE="$HF_HOME_DIR/hub"
export TRANSFORMERS_CACHE="$HF_HOME_DIR"
export TOKENIZERS_PARALLELISM=false
EOF

export HF_HOME="$HF_HOME_DIR"
export HF_HUB_CACHE="$HF_HOME_DIR/hub"
export TRANSFORMERS_CACHE="$HF_HOME_DIR"
export TOKENIZERS_PARALLELISM=false

log "Running GPU and import checks..."
python - <<'PY'
import torch
print("Python OK")
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
print("Parler-TTS imports OK")
PY

log "Checking Hugging Face login status..."
if hf auth whoami >/dev/null 2>&1; then
  hf auth whoami || true
else
  echo "Not logged in to Hugging Face yet."
  echo "IMPORTANT: ai4bharat/indic-parler-tts requires accepting model terms on Hugging Face."
  echo "Step 1: Open https://huggingface.co/ai4bharat/indic-parler-tts and accept access conditions."
  echo "Step 2: Run: source $PIPELINE_ROOT/scripts/activate_indic_parler_tts.sh && hf auth login"
fi

log "Setup complete."
echo "Activate with: source $PIPELINE_ROOT/scripts/activate_indic_parler_tts.sh"
echo "Production script path should be: $PIPELINE_ROOT/scripts/05_tts_indic_parler_production.py"
