#!/usr/bin/env python3
import sys, traceback, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.hf_client import HuggingFaceClient
from src.assistant_filing import AssistantFiling
from src.memory_tail import MemoryTail
from src.memory_store import FileMemory

print('Beginning Tesslate snapshot/download + load (may take many minutes and ~20GB).')
try:
    c = HuggingFaceClient('Tesslate/OmniCoder-9B', quantize=False)
    print('Model loaded; running generation...')
    out = c.run('Say hello in one sentence.', max_tokens=32, temperature=0.0)
    print('---OUTPUT---')
    print(out)
    status = 'success'
except Exception as e:
    traceback.print_exc()
    print('ERROR', e)
    status = 'error'

assistant = AssistantFiling()
short = f'Tesslate load {status}'
assistant.append_short_progress(short, label='tesslate')
tail = MemoryTail('~/ObsidianVaults/ProjectsVault/WorkingMemory.md')
tail.append(short, source='system')
try:
    tail.rollup_to_file()
except Exception:
    pass
mem = FileMemory('~/ObsidianVaults/ProjectsVault/WorkingMemory.md')
mem.append_snapshot('tesslate_load', short)
print('Logged short result to Obsidian.')
