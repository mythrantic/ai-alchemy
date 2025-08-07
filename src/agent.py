
import asyncio
import aiohttp
import os

from langchain.callbacks.base import BaseCallbackHandler, AsyncCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import ConfigurableField
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from pydantic import BaseModel, SecretStr

GROQ_API_KEY = SecretStr(os.getenv("GROQ_API_KEY", ""))
OLLLAMA_API_URL = SecretStr(os.getenv("OLLLAMA_API_URL", "http://localhost:11434"))
MODEL_NAME = SecretStr(os.getenv("MODEL_NAME", "llama3.1"))
SERPAPI_API_KEY = SecretStr(os.getenv("SERPAPI_API_KEY", ""))



# class config to help switch between Ollama and Groq, and more
class AgentConfig(BaseModel):
    model: str = "groq"  # or "ollama"
    ollama_api_url: str = OLLLAMA_API_URL.get_secret_value()
    model_name: str = MODEL_NAME.get_secret_value()
    serpapi_api_key: str = SERPAPI_API_KEY.get_secret_value()
    groq_api_key: str = GROQ_API_KEY.get_secret_value()
    supported_models: list[str] = ["groq", "ollama"]

    class Config:
        arbitrary_types_allowed = True
        
def get_llm(config: AgentConfig):
    if config.model == "groq":
        llm = ChatGroq(
            model=config.model_name,
            api_key=config.groq_api_key,
            temperature=0.0,
            streaming=True
        ).configurable_fields(
            callbacks=ConfigurableField(
                id="callbacks",
                name="Callbacks",
                description="List of callbacks to use for streaming",
            )
        )
    elif config.model == "ollama":
        llm = ChatOllama(
            model=config.model_name,
            api_url=config.ollama_api_url,
        )
    else:
        raise ValueError(f"Unsupported model: {config.model}. Supported models are: {config.supported_models}")
    return llm

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", (
            "You are a helpful AI assistant. When awering a user's question, "
            "you should first use one of the tools provided. Fster using a "
            "tool the tool output will be provided back to you. When you have "
            "all the information you need, you MUST use the final_answer tool "
            "to provide a final answer to the user. Use tools to answer the "
            "users's CURRENT question, not previous questions."
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# we use the article object for parsing serpapi results later
class Article(BaseModel):
    title: str
    source: str
    link: str
    snippet: str
    
    @classmethod
    def from_serpapi_result(cls, result: dict) -> "Article":
        return cls(
            title=result.get("title", ""),
            source=result.get("source", ""),
            link=result.get("link", ""),
            snippet=result.get("snippet", ""),
        )

# Tools Definition 
## we need to define all toolsa as async to simplify later code, but only the serpapi tool is actually async
@tool
async def add(x: float, y: float) -> float:
    """Adds two numbers together. add 'x' and 'y' to get the sum."""
    return x + y

@tool
async def multiply(x: float, y: float) -> float:
    """Multiplies two numbers together. multiply 'x' and 'y' to get the product."""
    return x * y

@tool
async def exponentiate(x: float, y: float) -> float:
    """Raises 'x' to the power of 'y'. exponentiate 'x' and 'y' to get the result."""
    return x ** y

@tool
async def subtract(x: float, y: float) -> float:
    """Subtracts 'y' from 'x'. subtract 'x' and 'y' to get the difference."""
    return x - y

@tool
async def divide(x: float, y: float) -> float:
    """Divides 'x' by 'y'. divide 'x' and 'y' to get the quotient."""
    if y == 0:
        raise ValueError("Cannot divide by zero.")
    return x / y

@tool
async def serpapi_search(query: str) -> list[Article]:
    """USe this tool to search the web"""
    params = {
        "q": query,
        "engine": "google",
        "api_key": SERPAPI_API_KEY.get_secret_value()
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(
            "https://serpapi.com/search", params=params
        ) as response:
            results = await response.json()
        
    return [
        Article.from_serpapi_result(result)
        for result in results.get("organic_results")
    ]

@tool
async def final_answer(answer: str, tools_used: list[str]) -> dict[str, str | list[str]]:
    """USe this tool to provide a final answer to the user."""
    return {
        "answer": answer,
        "tools_used": tools_used
    }

tools = [
    add,
    multiply,
    exponentiate,
    subtract,
    divide,
    serpapi_search,
    final_answer
]

# N/B! when we use sync tools we use tool.func, when async we use tool.coroutine. This is why we are making all tools async cause we can just do tool.coroutine. instead of checking if tool is sync use tool.func or if async use tool.coroutine
name2tool = {tool.name: tool.coroutine for tool in tools}

# Streaming Handler
class QueueCallbackHandler(AsyncCallbackHandler):
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue
        self.final_answer_seen = False
        
    async def __aiter__(self):
        while True:
            if self.queue.empty():
                await asyncio.sleep(0.1)
                continue
            token_or_done = await self.queue.get()
            if token_or_done == "<<DONE>>":
                return
            if token_or_done:
                yield token_or_done
                
            
                    
def agent_executor():
    # Placeholder for the agent executor function
    pass