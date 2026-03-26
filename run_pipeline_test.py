#!/usr/bin/env python3
"""Simple sandbox test for the RAG pipeline using local qwen3.5:9b via Ollama."""
from src.pipeline import RAGPipeline
import os

# ensure resources dir exists and add a tiny doc
res_dir = os.path.expanduser('~/ObsidianVaults/ProjectsVault/Resources')
os.makedirs(res_dir, exist_ok=True)
with open(os.path.join(res_dir, 'sample.txt'), 'w', encoding='utf-8') as f:
    f.write('Project Alpha: build a prototype for testing local LLM pipeline. Owners: you. Status: in progress. Next: run integration tests.\n')

p = RAGPipeline()
q = 'What are the next actions for Project Alpha?'
print('Running pipeline test (query -> qwen3.5:9b)')
resp = p.ask(q, k=2, max_tokens=300)
print('\n--- Response ---\n')
print(resp)

print('\nCheck WorkingMemory.md for appended snapshot.')
