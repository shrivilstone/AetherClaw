import os
import glob
from duckduckgo_search import DDGS
import json

import time

# Obsidian Integration Tools
VAULT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

_vault_cache = {"notes": [], "time": 0}
CACHE_TTL = 30 # seconds

def _get_notes():
    now = time.time()
    if now - _vault_cache["time"] < CACHE_TTL:
        return _vault_cache["notes"]
    notes = glob.glob(os.path.join(VAULT_ROOT, "**/*.md"), recursive=True)
    _vault_cache["notes"] = notes
    _vault_cache["time"] = now
    return notes

def list_obsidian_notes() -> str:
    """Lists all markdown notes in the Obsidian vault."""
    notes = _get_notes()
    relative_notes = [os.path.relpath(n, VAULT_ROOT) for n in notes if os.path.isfile(n)]
    return json.dumps(relative_notes)

def read_obsidian_note(note_path: str) -> str:
    """Reads the content of a specific note in the Obsidian vault."""
    full_path = os.path.join(VAULT_ROOT, note_path)
    if not os.path.exists(full_path) or not os.path.isfile(full_path):
        return f"Error: Note '{note_path}' not found or is not a file."
    with open(full_path, 'r', encoding='utf-8') as f:
        return f.read()

def search_obsidian_vault(query: str) -> str:
    """Searches the entire Obsidian vault for a specific query string."""
    notes = _get_notes()
    results = []
    for note in notes:
        if os.path.isfile(note):
            try:
                with open(note, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if query.lower() in content.lower():
                        results.append(os.path.relpath(note, VAULT_ROOT))
            except Exception:
                continue
    return json.dumps(results[:10])  # Limit results

def web_search(query: str, max_results: int = 5) -> str:
    """
    Perform a web search using DuckDuckGo.
    
    Args:
        query: The search query string.
        max_results: Maximum number of results to return.
        
    Returns:
        JSON string of the search results containing 'title', 'href', and 'body'.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return json.dumps([{"error": "No results found for query."}])
            return json.dumps(results)
    except Exception as e:
        return json.dumps([{"error": f"Search failed: {str(e)}"}])

# Define the unified tools schema to pass to Ollama's Chat API
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Perform a web search to find current information, news, or facts from the internet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The specific topic or question to search the web for."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_obsidian_notes",
            "description": "Lists all markdown notes available in your Obsidian vault root.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_obsidian_note",
            "description": "Reads the text content of a single Obsidian note by its relative path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note_path": {
                        "type": "string",
                        "description": "Relative path to the .md note (e.g. 'Projects/ProjectA.md')."
                    }
                },
                "required": ["note_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_obsidian_vault",
            "description": "Searches for a keyword or phrase across all notes in your vault.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search term to find in your notes."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# Dispatcher dictionary for quick execution
TOOL_FUNCTIONS = {
    "web_search": web_search,
    "list_obsidian_notes": list_obsidian_notes,
    "read_obsidian_note": read_obsidian_note,
    "search_obsidian_vault": search_obsidian_vault
}

if __name__ == "__main__":
    # Test the tool
    res = web_search("Latest advances in local LLMs", max_results=2)
    print("Test result:", res)
