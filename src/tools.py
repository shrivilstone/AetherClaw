from duckduckgo_search import DDGS
import json

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
    }
]

# Dispatcher dictionary for quick execution
TOOL_FUNCTIONS = {
    "web_search": web_search
}

if __name__ == "__main__":
    # Test the tool
    res = web_search("Latest advances in local LLMs", max_results=2)
    print("Test result:", res)
