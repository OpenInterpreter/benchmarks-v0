import os
import subprocess


if __name__ == "__main__":
    task = os.getenv("TASK")
    if task is None:
        raise ValueError("TASK is None!")
    
    display = os.getenv("DISPLAY_NUM")
    if display is None:
        raise ValueError("DISPLAY_NUM is None!")

    subprocess.run(
        ["/bin/bash", "-c", "echo $TASK | i -y"],
        env={
            **os.environ.copy(),
            "DISPLAY": f":1",
        },
    )
