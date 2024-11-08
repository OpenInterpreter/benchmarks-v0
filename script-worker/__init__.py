import os
from pathlib import Path
import subprocess
import traceback
from typing import Any, Dict, List, cast
from interpreter import OpenInterpreter


OUTPUT_PATH = Path("output/messages.json")
INPUT_PATH = Path("input")


def command_to_interpreter(cmd: Dict[str, Any]) -> OpenInterpreter:
    from .profile import interpreter

    interpreter.llm.model = cmd.get("model", interpreter.llm.model)  # type: ignore
    interpreter.llm.context_window = cmd.get("context_window", interpreter.llm.context_window)  # type: ignore
    interpreter.llm.api_base = cmd.get("api_base", interpreter.llm.api_base)  # type: ignore
    interpreter.llm.api_key = cmd.get("api_key", interpreter.llm.api_key)  # type: ignore
    interpreter.auto_run = cmd.get("auto_run", interpreter.auto_run)  # type: ignore
    interpreter.os = cmd.get("os_mode", interpreter.os)  # type: ignore
    interpreter.custom_instructions = cmd.get("custom_instructions", interpreter.custom_instructions)  # type: ignore
    interpreter.llm.supports_functions = cmd.get("supports_functions", interpreter.llm.supports_functions)
    return interpreter


def run(command: Dict[str, Any], prompt: str) -> List[Dict[str, str]]:
    assert INPUT_PATH.exists(), "input folder doesn't exist!"
    interpreter = command_to_interpreter(command)
    output = []

    try:
        output = cast(List, interpreter.chat(prompt, display=True, stream=False))
    except KeyboardInterrupt:
        output = [*interpreter.messages, { "role": "error", "content": "KeyboardInterrupt" }]
    except Exception as e:
        trace = traceback.format_exc()
        output = [*interpreter.messages, { "role": "error", "content": trace }]
    finally:
        interpreter.computer.terminate()
        return output


def run_script(script_path: Path, prompt: str, reqs_path: Path | None) -> List[Dict[str, str]]:
    """
    We expect the script's requirements to have been installed.
    """

    assert INPUT_PATH.exists(), "input folder doesn't exist!"

    compound_command = f"python3 {str(script_path)}"

    if reqs_path is not None:
        compound_command = f"python3 -m pip install -r {str(reqs_path)} && python3 {str(script_path)}"
    
    print("compound_command:", compound_command)

    env = os.environ.copy()
    p = subprocess.Popen(
        # ["python3", "-m", "pip", "install", "-r", str(reqs_path)],
        ["bash", "-c", compound_command],
        # cwd=str(INPUT_PATH),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={
            **env,
            "TASK": prompt,
        },
    )
    assert p.stdout is not None, "process stdout is None for some reason!"

    while p.poll() is None:
        output = p.stdout.readline().decode('utf-8')
        print(output)

    final_output = p.stdout.read().decode('utf-8')
    print(final_output)

    return []  # Where do the messages from??
    