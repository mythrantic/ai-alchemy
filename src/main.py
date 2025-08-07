from typing import Union
import asyncio

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

async def token_generator(content: str, streamer):
    pass
class QueueCallbackHandler:
    pass

@app.post("/api/invoke")
async def invoke(content: str):
    queue: asyncio.Queue[Union[str, bytes]] = asyncio.Queue()
    streamer = QueueCallbackHandler(queue)
    
    return StreamingResponse(
        token_generator(content, streamer),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache",
                 "Connection": "keep-alive"}
    )   
    
