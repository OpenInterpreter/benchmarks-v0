import json
import os
from pathlib import Path
import sys
import jsonschema
import pprint

from . import run, OUTPUT_PATH


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

    if len(sys.argv) != 4:
        print("Usage: python -m worker.run <command:json-str> <prompt:str> <output-dir:str>")
        exit(1)

    # grabbing command.
    command_config = sys.argv[1]
    command_config_as_json = json.loads(command_config)
    jsonschema.validate(instance=command_config_as_json, schema=schema)

    # grabbing prompt.
    prompt = sys.argv[2]

    # grabbing output directory.
    out_dir = Path(sys.argv[3])
    print("output dir:", out_dir)

    print("command config:")
    pprint.pp(command_config_as_json)
    print("prompt:", prompt)

    # running!
    full_output_path = OUTPUT_PATH
    messages = run(command_config_as_json, prompt)
    if not full_output_path.parent.exists():
        os.makedirs(full_output_path.parent)
    with open(full_output_path, "w+") as f:
        json.dump(messages, f)
