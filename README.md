# AetherClaw

A premium web-based AI Command Center inspired by OpenClaw, built on top of a local RAG + working-memory pipeline using Ollama (`qwen3.5:9b`).

## Features
- **Intelligent RAG**: Hybrid retrieval (Dense FAISS + BM25) for high-quality context.
- **Working Memory**: Real-time synchronization with Obsidian snapshots.
- **Premium Dashboard**: Glassmorphism UI with dark mode, system stats, and interactive chat.
- **System Monitoring**: Live tracking of CPU, RAM, and Disk usage via the web interface.

## Quick Start

### Prerequisites
- Ollama installed and running with `qwen3.5:9b`.
- Python 3.8+ and virtual environment.

### Installation
1. Activate your environment and install dependencies:
```bash
source ai_env/bin/activate
pip install -r requirements.txt
```

2. Rebuild the FAISS index from your Obsidian resources:
```bash
python scripts/rebuild_index.py --resources ~/ObsidianVaults/ProjectsVault/Resources
```

3. Launch the AetherClaw Server:
```bash
python server.py
```

4. Access the Dashboard:
Open your browser and navigate to `http://localhost:8000`.

## Architecture
- **Backend**: FastAPI server wrapping the RAG pipeline.
- **Frontend**: Vanilla HTML/JS with premium CSS aesthetics.
- **Inference**: Ollama CLI (`qwen3.5:9b`).
- **Memory**: Markdown-based persistent memory in Obsidian.

---
*Inspired by OpenClaw. Customized for AetherClaw.*
