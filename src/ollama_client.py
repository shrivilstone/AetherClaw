import requests
from typing import Optional
from .tools import TOOLS_SCHEMA, TOOL_FUNCTIONS

class OllamaClient:
    """Wrapper to call local Ollama model via REST API to support native Tool Calling.
    """
    def __init__(self, model: str = "hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M"):
        self.model = model
        self.url = "http://localhost:11434/api/chat"

    def run(self, prompt: str, max_tokens: Optional[int] = None, temperature: Optional[float]=None, timeout: int = 600) -> str:
        messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "tools": TOOLS_SCHEMA
        }

        try:
            response = requests.post(self.url, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            message = data.get("message", {})
            
            # Check if tools were called
            tool_calls = message.get("tool_calls")
            if tool_calls:
                # Add assistant's tool call request to history
                messages.append(message)
                
                # Execute tools and append results
                for tool in tool_calls:
                    func_name = tool['function']['name']
                    args = tool['function']['arguments']
                    if func_name in TOOL_FUNCTIONS:
                        print(f"⚡ AetherClaw is executing Tool: {func_name} with {args}...")
                        result = TOOL_FUNCTIONS[func_name](**args)
                        messages.append({
                            "role": "tool",
                            "content": result,
                            "name": func_name
                        })
                
                # Second inference step with tool results
                payload["messages"] = messages
                # Remove tools for final response to prevent infinite loops
                payload.pop("tools", None)
                
                response2 = requests.post(self.url, json=payload, timeout=timeout)
                response2.raise_for_status()
                data2 = response2.json()
                return data2.get("message", {}).get("content", "").strip()
            
            return message.get("content", "").strip()
            
        except Exception as e:
            raise RuntimeError(f"Ollama API request failed: {e}")

if __name__ == '__main__':
    c = OllamaClient()
    print(c.run("What is the capital of Paris?"))
