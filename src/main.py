import asyncio
from src.agent import agent_executor, QueueCallbackHandler

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import debugpy

app = FastAPI()
debugpy.listen(("0.0.0.0", 5678))

# Enable CORS for all/specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Welcome": "to the AI Alchemy API"}

async def token_generator(content: str, streamer: QueueCallbackHandler):
    task = asyncio.create_task(agent_executor.invoke(
        input=content,
        streamer=streamer,
        verbose=True 
    ))
    
    async for token in streamer:
        try:
            if token == "<<STEP_END>>":
                # endo of step token
                yield "</step>"
            elif tool_calls := token.message.additional_kwargs.get("tool_calls"):
                if tool_name := tool_calls[0]["function"]["name"]:
                    # send start of step token followed by step name tokens
                    yield f"<step><step_name>{tool_name}</step_name>"
                if tool_args := tool_calls[0]["function"].get("arguments"):
                    # tool argd are streamed directly, ensure its properly encoded
                    yield tool_args
        except Exception as e:
            print(f"Error in token generation: {e}")
            continue
    await task


@app.post("/api/invoke")
async def invoke(content: str):
    queue: asyncio.Queue = asyncio.Queue()
    streamer = QueueCallbackHandler(queue)
    
    return StreamingResponse(
        token_generator(content, streamer),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache",
                 "Connection": "keep-alive"}
    )   
    
