
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
        )
    elif config.model == "ollama":
        llm = ChatOllama(
            model=config.model_name,
            api_url=config.ollama_api_url,
            temperature=0.0,
            streaming=True,
        )
    else:
        raise ValueError(f"Unsupported model: {config.model}. Supported models are: {config.supported_models}")
    return llm.configurable_fields(
            callbacks=ConfigurableField(
                id="callbacks",
                name="Callbacks",
                description="List of callbacks to use for streaming",
            )
        )

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
                
    async def on_llm_new_token(self, *args, **kwargs) -> None:
        chunk = kwargs.get("chunk")
        if chunk and chunk.message.additional_kwargs.get("too_calls"):
            if chunk.message.additional_kwargs["tool_calls"][0]["function"]["name"] == "final_answer":
                self.final_answer_seen = True
        self.queue.put_nowait(kwargs.get("chunk"))
        
    async def on_llm_end(self, *args, **kwargs) -> None:
        if not self.final_answer_seen:
            self.queue.put_nowait("<<DONE>>")
        else:
            self.queue.put_nowait("<<STEP_END>>")
        
async def execute_tool(tool_call: AIMessage) -> ToolMessage:
    tool_name = tool_call.tool_calls[0]["name"]
    tool_args = tool_call.tool_calls[0]["args"]
    tool_out = await name2tool[tool_name](**tool_args)
    return ToolMessage(
        content=f"{tool_out}",
        tool_call_id=tool_call.tool_calls[0]["id"]
    )
    

# Agent Executor
class CustomAgentExecutor:
    def __init(self, max_iterations: int = 3):
        self.chat_history: list[BaseMessage] = []
        self.max_iterations = max_iterations
        self.agent = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x["chat_history"],
                "agent_scratchpad": lambda x: x.get("agent_scratchpad", [])
            }
            | prompt
            | get_llm(config=AgentConfig(model="groq")).bind_tools(tools, tool_choice="any")
            )
    async def invoke(self, input: str, streamer: QueueCallbackHandler, verbose: bool = False):
        """invoke the agent. but this is done iteretively in a loop until we reach the final answe"""
        count = 0
        final_answer: str | None = None
        agent_scratchpad: list[AIMessage | ToolMessage] = []
        
        # the streaming magic
        async def stream(query: str) -> list[AIMessage]:
            configured_agent = self.agent.with_config(
                callbacks=[streamer]
            )
            
            ## the output dict that we eill be populating with our streamed output
            outputs = []
            async for token in configured_agent.astream({
                "input": query,
                "chat_history": self.chat_history,
                "agent_scratchpad": agent_scratchpad
            }):
                tool_calls = token.additional_kwargs.get("tool_calls")
                if tool_calls:
                    
                    # first check if we have a tool call id  - this indicates neq tool
                    if tool_calls[o]["id"]:
                        outputs.append(token)
                    else:
                        # this is a tool call that we have already seen, so we just update the last token
                        outputs[-1] += token
                        
                else:
                    pass
            return [
                AIMessage(
                    content=x.content,
                    tool_calls=x.tool_calls,
                    tool_call_id=x.tool_calls[0]["id"]
                ) for x in outputs
            ]
        
        while count < self.max_iterations:
            ## invoke a step for the agent to generate a tool call
            tool_calls = await stream(query=input)
            
            ## gathr the tool execution coroutines
            tool_obs = await asyncio.gather(
                *[execute_tool(tool_call) for tool_call in tool_calls]
            )
            
            ## basictly the scrach pad we want is something like this ordered according to the tool calls id
            """
            !IMPORTANT! the order of tool calls and observations is important, but not the order of the tool call ids.
            the tool call ids just need to be together. it doesnt matter if the tool call id 2 is before tool call id 1, as long as the tool call id 2 is followed by its observation
            [
                
                AIMessage(content="...", tool_calls=[{"id": "1", "name": "search", "args": {"query": "..."}}]),
                ToolMessage(content="...", tool_call_id="1"),
                AIMessage(content="...", tool_calls=[{"id": "2", "name": "add", "args": {"x": 1, "y": 2}}]),
                ToolMessage(content="3", tool_call_id="2"),
            ]
            """
            
            ## appemd tool calls and observations to the agent scratchpad in order
            id2tool_obs = {tool_call.tool_call_id: tool_obs for tool_call, tool_obs in zip(tool_calls, tool_obs)}
            
            for tool_call in tool_calls:
                agent_scratchpad.extend([
                    tool_call, # AIMessage with tool call
                    id2tool_obs[tool_call.tool_call_id] # ToolMessage with tool observation
                ])
                
            count += 1
            ## check if we have a final answer
            found_final_answer = False
            for tool_call in tool_calls:
                if tool_call.tool_calls[0]["name"] == "final_answer":
                    final_answer_call = tool_call.tool_calls[0]
                    final_answer =  final_answer_call["args"]["answer"]
                    found_final_answer = True
                    break
                
            ## Only break the loop if we have a final answer
            if found_final_answer:
                break
        
        ## add the final output to the chat history, only the answer field
        self.chat_history.extend({
            HumanMessage(content=input),
            AIMessage(content=final_answer if final_answer else "No final answer found")
        })
        
        return final_answer_call if final_answer else {"answer": "No final answer found", "tools_used": []}
    

# Initi the agent executor
agent_executor = CustomAgentExecutor(max_iterations=3)