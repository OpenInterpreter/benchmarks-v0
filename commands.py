import os
from typing import Dict, NotRequired, TypedDict


# Need to figure out how to parameterize the hostname based on the runner!
hostname = f"host.docker.internal"

openapi_key = os.getenv("OPENAI_API_KEY", "")
custom_instructions = """I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER].
YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
If you are asked for a number, don’t use comma to write your number neither use units such as $ or percent sign unless specified otherwise.
If you are asked for a string, don’t use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.
If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
"""


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
        "custom_instructions": custom_instructions
    },
    "gpt4turbo": {
        "auto_run": True,
        "api_key": openapi_key,
        "custom_instructions": custom_instructions,
        "model": "gpt-4-turbo",
        "context_window": 128000,
    },
    "gpt4o": {
        "auto_run": True,
        "model": "openai/gpt-4o",
        "context_window": 128000,
        "api_key": openapi_key,
        "custom_instructions": custom_instructions
    },
    "gpt4": {
        "auto_run": True,
        "model": "openai/gpt-4",
        "context_window": 8192,
        "api_key": openapi_key,
        "custom_instructions": custom_instructions
    },
    "gpt35turbo": {
        "auto_run": True,
        "model": "openai/gpt-3.5-turbo-0125",
        "context_window": 16385,
        "api_key": openapi_key,
        "custom_instructions": custom_instructions
    },
    "llama3": {
        "auto_run": True,
        "model": "ollama/llama3",
        "context_window": 2048,
        "api_base": f"http://{hostname}:11434",
        "custom_instructions": custom_instructions
    },
}
