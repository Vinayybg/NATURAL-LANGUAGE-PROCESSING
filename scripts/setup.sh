#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
#  scripts/setup.sh — One-command local environment setup
#  Usage:  bash scripts/setup.sh
# ═══════════════════════════════════════════════════════════
set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }

log "NVIDIA Strategic Intelligence Agent — Setup"
echo "─────────────────────────────────────────────────"

# 1. Python version check
PYTHON=$(command -v python3.12 || command -v python3.11 || command -v python3)
PY_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
log "Python: $PYTHON ($PY_VERSION)"

# 2. Create virtual environment
if [ ! -d "venv" ]; then
    log "Creating virtual environment…"
    $PYTHON -m venv venv
fi

# Activate
source venv/bin/activate
log "Virtual environment activated"

# 3. Upgrade pip silently
pip install --upgrade pip --quiet

# 4. Install dependencies
log "Installing dependencies (may take 5-10 minutes on first run)…"
pip install -r requirements.txt --quiet

# 5. Create data directories
mkdir -p data logs
log "Created data/ and logs/ directories"

# 6. Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    cp .env.example .env
    warn ".env created from template — fill in your GROQ_API_KEY before running"
else
    log ".env already exists — skipping"
fi

echo ""
log "✅ Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Edit .env and add your GROQ_API_KEY (free at https://console.groq.com)"
echo "  2. Run the pipeline:   python pipeline.py --all --report"
echo "  3. Launch dashboard:   streamlit run app.py"
echo ""
