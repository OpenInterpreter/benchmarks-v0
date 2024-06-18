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

    if len(sys.argv) != 3:
        print("given args:\n", "\n--------\n".join(sys.argv))
        print("Usage: python -m worker.run <command:json-str> <cwd:str>")
        exit(1)

    # grabbing command.
    command_config = sys.argv[1]
    command_config_as_json = json.loads(command_config)
    jsonschema.validate(instance=command_config_as_json, schema=schema)

    # grabbing the current working directory this script is to use.
    cwd = sys.argv[2]
    os.chdir(cwd)
    print("cwd:", cwd)

    run(command_config_as_json)
