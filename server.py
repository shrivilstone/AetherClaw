from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
import psutil
from src.pipeline import RAGPipeline

app = FastAPI(title="AetherClaw API")

# Global pipeline instance
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        print("Initializing RAG Pipeline (this may take a moment)...")
        _pipeline = RAGPipeline()
        print("RAG Pipeline initialized.")
    return _pipeline

class ChatRequest(BaseModel):
    query: str
    k: int = 3
    max_tokens: int = 300

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        pipeline = get_pipeline()
        response = pipeline.ask(request.query, k=request.k, max_tokens=request.max_tokens)
        return {"response": response}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def stats():
    try:
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        disk = psutil.disk_usage('/').percent
        
        # Grab active pipeline model definition
        pipeline = get_pipeline()
        active_model = getattr(pipeline.client, 'model', 'unknown')
        
        return {
            "cpu": cpu,
            "ram": ram,
            "disk": disk,
            "model": active_model,
            "status": "Healthy"
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/memory")
async def memory():
    try:
        working_memory_path = os.path.expanduser('~/ObsidianVaults/ProjectsVault/WorkingMemory.md')
        if os.path.exists(working_memory_path):
            with open(working_memory_path, 'r') as f:
                content = f.read()
            return {"content": content}
        else:
            return {"content": "WorkingMemory.md not found."}
    except Exception as e:
        return {"error": str(e)}

# Serve static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # Use reload=True for development if needed
    uvicorn.run(app, host="0.0.0.0", port=8000)
