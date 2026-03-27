import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import json

class MCPClient:
    def __init__(self, server_params: StdioServerParameters):
        self.server_params = server_params
        self.session = None
        self._client_context = None

    async def connect(self):
        self._client_context = stdio_client(self.server_params)
        read, write = await self._client_context.__aenter__()
        self.session = ClientSession(read, write)
        await self.session.__aenter__()
        await self.session.initialize()

    async def get_tools(self):
        if not self.session:
            return []
        tools_resp = await self.session.list_tools()
        # Convert MCP tools to Ollama-compatible TOOLS_SCHEMA
        ollama_tools = []
        for tool in tools_resp.tools:
            ollama_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            })
        return ollama_tools

    async def call_tool(self, name: str, arguments: dict):
        if not self.session:
            return "Error: MCP session not initialized."
        result = await self.session.call_tool(name, arguments)
        return result.content

    async def disconnect(self):
        if self.session:
            await self.session.__aexit__(None, None, None)
        if self._client_context:
            await self._client_context.__aexit__(None, None, None)

async def main():
    # Example testing code
    params = StdioServerParameters(
        command="python3",
        args=["-m", "mcp_server_example"] # This is just a placeholder
    )
    client = MCPClient(params)
    try:
        await client.connect()
        tools = await client.get_tools()
        print("Discovered tools:", json.dumps(tools, indent=2))
    except Exception as e:
        print(f"MCP Connection failed (expected if no server is running): {e}")
    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
