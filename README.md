# Self-Extending AI System

An LLM agent that generates its own Python tools on demand using the
**Model Context Protocol (MCP)**. The agent can create, register, invoke,
inspect, and remove tools at runtime — all within a single process.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    main.py (entry point)                │
│                                                         │
│  ┌──────────────┐    anyio     ┌──────────────────────┐ │
│  │  SelfExtend- │◄────────────►│  DynamicMCPServer    │ │
│  │  ingAgent    │  mem streams │  (mcp_server.py)     │ │
│  │  (agent.py)  │              │                      │ │
│  └──────────────┘              │  ┌────────────────┐  │ │
│         │                      │  │ DynamicTool-   │  │ │
│         │ OpenAI API           │  │ Registry       │  │ │
│         ▼                      │  │ (tool_registry)│  │ │
│    LLM (Ollama /               │  └────────────────┘  │ │
│    OpenAI / etc.)              │  ┌────────────────┐  │ │
│                                │  │ CodeGeneration-│  │ │
│                                │  │ Pipeline       │  │ │
│                                │  │ (code_gen...)  │  │ │
│                                │  └────────────────┘  │ │
│                                └──────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Requirements

- Python ≥ 3.11
- A running LLM endpoint (Ollama, OpenAI, LM Studio, etc.)

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/ms1963/Codegen.git
cd codegen

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your LLM settings
```

## Running

### Interactive mode (default)

```bash
cd codegen
python main.py
```

### Single query mode

```bash
cd codegen
python main.py --query "Calculate the compound interest on $10,000 at 5% for 10 years"
```

### With custom LLM settings

```bash
cd codegen
python main.py \
  --base-url https://api.openai.com/v1 \
  --api-key sk-... \
  --model gpt-4o \
  --log-level DEBUG
```

### With Docker

```bash
# Build
docker build -t self-extending-ai .

# Run interactive (with local Ollama)
docker run -it \
  -e LLM_BASE_URL=http://host.docker.internal:11434/v1 \
  -e LLM_MODEL=qwen2.5:14b \
  -v$(pwd)/data:/data \
  self-extending-ai

# Run single query
docker run --rm \
  -e LLM_BASE_URL=http://host.docker.internal:11434/v1 \
  -v $(pwd)/data:/data \
  self-extending-ai --query "What is the square root of 144?"
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:11434/v1` | LLM API base URL |
| `LLM_API_KEY` | `ollama` | API key |
| `LLM_MODEL` | `qwen2.5:14b` | Model name |
| `TOOLS_FILE` | `tools.json` | Tool persistence file |
| `LOG_LEVEL` | `INFO` | Logging level |

## Built-in Meta-Tools

| Tool | Description |
|---|---|
| `generate_and_register_tool` | Generate a new Python tool from a description |
| `list_registered_tools` | List all currently registered dynamic tools |
| `get_tool_source` | Inspect the source code of any registered tool |
| `remove_tool` | Remove a tool from the registry |
| `get_registry_stats` | Get usage statistics for all tools |

## Interactive Commands

| Command | Action |
|---|---|
| `exit` / `quit` | Stop the system |
| `clear` | Reset conversation history |
| `stats` | Show registry statistics |

## Issues Fixed (30 total)

| Severity | Count | Examples |
|---|---|---|
| 🔴 Critical | 10 | Deadlocks, missing functions, deprecated APIs |
| 🟡 Medium | 13 | Edge cases, type handling, error loops |
| 🟢 Low | 7 | Consistency, persistence, minor bugs |
