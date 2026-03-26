import subprocess
import shlex
from typing import Optional

class OllamaClient:
    """Minimal wrapper to call local Ollama model (qwen3.5:9b by default).

    Uses the `ollama` CLI (falls back to HTTP if needed). Captures stdout.
    """
    def __init__(self, model: str = "qwen3.5:9b"):
        self.model = model

    def run(self, prompt: str, max_tokens: Optional[int] = None, temperature: Optional[float]=None, timeout: int = 600) -> str:
        # Use the Ollama CLI and pass the prompt via stdin (works with local Ollama).
        # Ollama CLI accepts the prompt as a positional argument: `ollama run MODEL "prompt"`.
        # The local Ollama CLI in this environment doesn't support --max-tokens or --temperature.
        cmd = ["ollama", "run", self.model, prompt]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = proc.stdout.strip()
        if proc.returncode != 0:
            # include stderr for debugging
            raise RuntimeError(f"ollama run failed: {proc.returncode}: {proc.stderr.strip()}")
        return out

if __name__ == '__main__':
    c = OllamaClient()
    print(c.run("Say hello in one sentence."))
