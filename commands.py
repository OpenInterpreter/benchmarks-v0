import os
from typing import Dict
from benchmark import OpenInterpreterCommand


openapi_key = os.getenv("OPENAI_API_KEY", "")
custom_instructions = """I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER].
YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
If you are asked for a number, don’t use comma to write your number neither use units such as $ or percent sign unless specified otherwise.
If you are asked for a string, don’t use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.
If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
"""


commands: Dict[str, OpenInterpreterCommand] = {
    "oai_default": {
        "auto_run": True,
        "api_key": openapi_key,
        "custom_instructions": custom_instructions
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
        "api_base": "http://192.168.1.86:11434",
        "custom_instructions": custom_instructions
    },
}

