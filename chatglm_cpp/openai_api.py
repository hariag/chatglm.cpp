import asyncio
import json
import logging
import time
from typing import Dict, List, Literal, Optional, Union

import chatglm_cpp
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, computed_field
from pydantic_settings import BaseSettings
from sse_starlette.sse import EventSourceResponse

logging.basicConfig(level=logging.INFO, format=r"%(asctime)s - %(module)s - %(levelname)s - %(message)s")
MODEL_PATH="/local/llm_models/chatglm-ggml_q8_0.bin"
MODEL_HASH = None
PREFIX = "/llmapi"

def calculate_sha256(filename):
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(blksize), b""):
            hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


class Settings(BaseSettings):
    model: str = "chatglm3-ggml.bin"
    num_threads: int = 0


class ToolCallFunction(BaseModel):
    arguments: str
    name: str


class ToolCall(BaseModel):
    function: Optional[ToolCallFunction] = None
    type: Literal["function"]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    tool_calls: Optional[List[ToolCall]] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal["system", "user", "assistant"]] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class ChatCompletionToolFunction(BaseModel):
    description: Optional[str] = None
    name: str
    parameters: Dict


class ChatCompletionTool(BaseModel):
    type: Literal["function"] = "function"
    function: ChatCompletionToolFunction


class ChatCompletionRequest(BaseModel):
    model: str = "default-model"
    messages: List[ChatMessage]
    temperature: float = Field(default=0.95, ge=0.0, le=2.0)
    top_k: int = Field(default=0, ge=0, le=1024)
    top_p: float = Field(default=0.7, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, ge=0.0, le=100.0)
    stream: bool = False
    max_tokens: int = Field(default=2048, ge=0)
    tools: Optional[List[ChatCompletionTool]] = None

    model_config = {
        "json_schema_extra": {"examples": [{"model": "default-model", "messages": [{"role": "user", "content": "你好"}]}]}
    }


class ChatCompletionResponseChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop", "length", "function_call"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int

    @computed_field
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ChatCompletionResponse(BaseModel):
    id: str = "chatcmpl"
    model: str = "default-model"
    object: Literal["chat.completion", "chat.completion.chunk"]
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: Union[List[ChatCompletionResponseChoice], List[ChatCompletionResponseStreamChoice]]
    usage: Optional[ChatCompletionUsage] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "chatcmpl",
                    "model": "default-model",
                    "object": "chat.completion",
                    "created": 1691166146,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 17, "completion_tokens": 29, "total_tokens": 46},
                }
            ]
        }
    }


settings = Settings()
app = FastAPI()
global MODEL_HASH
if not MODEL_HASH:
    MODEL_HASH = calculate_sha256(MODEL_PATH)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)
pipeline = chatglm_cpp.Pipeline(settings.model)
lock = asyncio.Lock()


def stream_chat(messages, body):
    yield ChatCompletionResponse(
        object="chat.completion.chunk",
        choices=[ChatCompletionResponseStreamChoice(delta=DeltaMessage(role="assistant"))],
    )

    for chunk in pipeline.chat(
        messages=messages,
        max_length=body.max_tokens,
        do_sample=body.temperature > 0,
        top_p=body.top_p,
        temperature=body.temperature,
        num_threads=settings.num_threads,
        stream=True,
    ):
        yield ChatCompletionResponse(
            object="chat.completion.chunk",
            choices=[ChatCompletionResponseStreamChoice(delta=DeltaMessage(content=chunk.content))],
        )

    yield ChatCompletionResponse(
        object="chat.completion.chunk",
        choices=[ChatCompletionResponseStreamChoice(delta=DeltaMessage(), finish_reason="stop")],
    )


async def stream_chat_event_publisher(history, body):
    output = ""
    try:
        async with lock:
            for chunk in stream_chat(history, body):
                await asyncio.sleep(0)  # yield control back to event loop for cancellation check
                output += chunk.choices[0].delta.content or ""
                yield chunk.model_dump_json(exclude_unset=True)
        logging.info(f'prompt: "{history[-1]}", stream response: "{output}"')
    except asyncio.CancelledError as e:
        logging.info(f'prompt: "{history[-1]}", stream response (partial): "{output}"')
        raise e


@app.post("{}/v1/chat/completions".format(PREFIX))
async def create_chat_completion(body: ChatCompletionRequest) -> ChatCompletionResponse:
    def to_json_arguments(arguments):
        def tool_call(**kwargs):
            return kwargs

        try:
            return json.dumps(eval(arguments, dict(tool_call=tool_call)))
        except Exception:
            return arguments

    if not body.messages:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty messages")

    messages = [chatglm_cpp.ChatMessage(role=msg.role, content=msg.content) for msg in body.messages]
    if body.tools:
        system_content = (
            "Answer the following questions as best as you can. You have access to the following tools:\n"
            + json.dumps([tool.model_dump() for tool in body.tools], indent=4)
        )
        messages.insert(0, chatglm_cpp.ChatMessage(role="system", content=system_content))

    if body.stream:
        generator = stream_chat_event_publisher(messages, body)
        return EventSourceResponse(generator)

    max_context_length = 512
    output = pipeline.chat(
        messages=messages,
        max_length=body.max_tokens,
        max_context_length=max_context_length,
        do_sample=body.temperature > 0,
        top_k=body.top_k,
        top_p=body.top_p,
        temperature=body.temperature,
        repetition_penalty=body.repetition_penalty,
    )
    logging.info(f'prompt: "{messages[-1].content}", sync response: "{output.content}"')
    prompt_tokens = len(pipeline.tokenizer.encode_messages(messages, max_context_length))
    completion_tokens = len(pipeline.tokenizer.encode(output.content, body.max_tokens))

    finish_reason = "stop"
    tool_calls = None
    if output.tool_calls:
        tool_calls = [
            ToolCall(
                type=tool_call.type,
                function=ToolCallFunction(
                    name=tool_call.function.name, arguments=to_json_arguments(tool_call.function.arguments)
                ),
            )
            for tool_call in output.tool_calls
        ]
        finish_reason = "function_call"

    return ChatCompletionResponse(
        object="chat.completion",
        choices=[
            ChatCompletionResponseChoice(
                message=ChatMessage(role="assistant", content=output.content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ],
        usage=ChatCompletionUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class ModelData(TypedDict):
    id: str
    object: Literal["model"]
    owned_by: str
    permissions: List[str]
    title: str
    model_name: str
    hash: str
    sha256: str
    filename: str
    config: dict


class ModelList(TypedDict):
    object: Literal["list"]
    data: List[ModelData]


@router.get("{}/v1/models".format(PREFIX), summary="Models")
async def get_models() -> ModelList:
    sha256 = MODEL_HASH
    if sha256:
        shorthash = sha256[:10]
    else:
        shorthash = ""
    file_name = MODEL_PATH
    title = "{} [{}]".format(os.path.basename(file_name), shorthash)
    model_name = Path(file_name).stem
    config = {}
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_PATH,
                "object": "model",
                "owned_by": "me",
                "permissions": [],
                "title": title,
                "model_name": model_name,
                "hash": shorthash,
                "sha256": sha256,
                "filename": file_name,
                "config": config
            }
        ],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
