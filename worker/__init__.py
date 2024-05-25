from pathlib import Path
import traceback
from typing import Any, Dict, List, cast
from interpreter import OpenInterpreter


OUTPUT_PATH = Path("output/messages.json")


def command_to_interpreter(cmd: Dict[str, Any]) -> OpenInterpreter:
    interpreter = OpenInterpreter(import_computer_api=True)
    interpreter.llm.model = cmd.get("model", interpreter.llm.model)  # type: ignore
    interpreter.llm.context_window = cmd.get("context_window", interpreter.llm.context_window)  # type: ignore
    interpreter.llm.api_base = cmd.get("api_base", interpreter.llm.api_base)  # type: ignore
    interpreter.llm.api_key = cmd.get("api_key", interpreter.llm.api_key)  # type: ignore
    interpreter.auto_run = cmd.get("auto_run", interpreter.auto_run)  # type: ignore
    interpreter.os = cmd.get("os_mode", interpreter.os)  # type: ignore
    interpreter.custom_instructions = cmd.get("custom_instructions", interpreter.custom_instructions)  # type: ignore
    return interpreter


def run(command: Dict[str, Any], prompt: str, display: bool) -> List[Dict[str, str]]:
    interpreter = command_to_interpreter(command)

    try:
        output = cast(List, interpreter.chat(prompt, display=display, stream=False))
    except KeyboardInterrupt:
        output = [*interpreter.messages, { "role": "error", "content": "KeyboardInterrupt" }]
    except Exception as e:
        trace = traceback.format_exc()
        output = [*interpreter.messages, { "role": "error", "content": trace }]
    finally:
        interpreter.computer.terminate()
        return output
    