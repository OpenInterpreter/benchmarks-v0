import argparse
import json
import os
from pathlib import Path
import sys
import jsonschema
import pprint

from . import run, OUTPUT_PATH, run_script


if __name__ == "__main__":
    schema = {
        "type": "object",
        "properties": {
            "auto_run": { "type": "boolean" },
            "os_mode": { "type": "boolean" },
            "model": { "type": "string" },
            "context_window": { "type": "number" },
            "api_base": { "type": "string" },
            "api_key": { "type": "string" },
            "custom_instructions": { "type": "string" }
        }
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str)
    parser.add_argument("prompt", type=str)
    parser.add_argument("cwd", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("script_path", nargs="?", type=str)  # If not None ignore command.
    parser.add_argument("requirements_path", nargs="?", type=str)

    # if len(sys.argv) != 5:
    #     print("Usage: python -m worker.run <command:json-str> <prompt:str> <cwd:str> <output-dir:str>")
    #     exit(1)

    args = parser.parse_args()

    # grabbing command.
    command_config = args.command
    command_config_as_json = json.loads(command_config)
    jsonschema.validate(instance=command_config_as_json, schema=schema)

    # grabbing prompt.
    prompt = args.prompt

    # grabbing the current working directory this script is to use.
    cwd = args.cwd
    os.chdir(cwd)

    # grabbing output directory.
    out_dir = Path(args.output_dir)
    print("output dir:", out_dir)

    print("command config:")
    pprint.pp(command_config_as_json)
    print("prompt:", prompt)

    print("agent path:", args.script_path)
    print("requirements path:", args.requirements_path)

    script_path = args.script_path
    full_output_path = OUTPUT_PATH
    messages = []

    if script_path is None:
        # running!
        print("Running without agent")
        messages = run(command_config_as_json, prompt)
    else:
        print("Running with agent")
        reqs_path = Path(args.requirements_path) if args.requirements_path is not None else None
        messages = run_script(Path(script_path), prompt, reqs_path)

    if not full_output_path.parent.exists():
        os.makedirs(full_output_path.parent)
    with open(full_output_path, "w+") as f:
        json.dump(messages, f)
