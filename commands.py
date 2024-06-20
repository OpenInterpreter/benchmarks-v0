import os
from typing import Dict, NotRequired, TypedDict


# Need to figure out how to parameterize the hostname based on the runner!
hostname = f"host.docker.internal"
openapi_key = os.getenv("OPENAI_API_KEY", "")


class OpenInterpreterCommand(TypedDict):
    auto_run: NotRequired[bool]
    os_mode: NotRequired[bool]
    model: NotRequired[str]
    context_window: NotRequired[int]
    api_base: NotRequired[str]
    api_key: NotRequired[str]
    custom_instructions: NotRequired[str]
    supports_functions: NotRequired[bool]


commands: Dict[str, OpenInterpreterCommand] = {
    "": {},
    "groq-llama3": {
        "model": "groq/llama3-8b-8192",
        "context_window": 8192,
        "supports_functions": False,
        "api_key": os.getenv("GROQ_API_KEY", ""),
        "auto_run": True
    },
    "oai_default": {
        "auto_run": True,
        "api_key": openapi_key,
    },
    "gpt4turbo": {
        "auto_run": True,
        "api_key": openapi_key,
        "model": "gpt-4-turbo",
        "context_window": 128000,
    },
    "gpt4o": {
        "auto_run": True,
        "model": "openai/gpt-4o",
        "context_window": 128000,
        "api_key": openapi_key,
    },
    "gpt4": {
        "auto_run": True,
        "model": "openai/gpt-4",
        "context_window": 8192,
        "api_key": openapi_key,
    },
    "gpt35turbo": {
        "auto_run": True,
        "model": "openai/gpt-3.5-turbo-0125",
        "context_window": 16385,
        "api_key": openapi_key,
    },
    "llama3": {
        "auto_run": True,
        "model": "ollama/llama3",
        "context_window": 2048,
        "api_base": f"http://{hostname}:11434",
    },
}
