# ── Prerequisites ────────────────────────────────────────────────────────────
# Python 3.11+
python --version

# ── Setup ────────────────────────────────────────────────────────────────────
git clone <repo-url> && cd self-extending-ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# ── Configure ────────────────────────────────────────────────────────────────
cp .env.example .env
# Edit .env: set LLM_BASE_URL, LLM_API_KEY, LLM_MODEL

# ── Run ──────────────────────────────────────────────────────────────────────
cd codegen
python main.py                          # interactive mode
python main.py -q "your question here" # single-query mode

# ── Docker ───────────────────────────────────────────────────────────────────
docker build -t self-extending-ai .
docker run -it -v $(pwd)/data:/data self-extending-ai
