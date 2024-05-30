import logging
import time

from dotenv import load_dotenv
from e2b_desktop import Desktop

load_dotenv()

logging.basicConfig(level=logging.INFO)

with Desktop() as desktop: # Using the `with` clause, the sandbox automatically calls `close()` on itself once we run all the code inside the clause.
    desktop.screenshot("screenshot-1.png")

    # Create file and open text editor
    file = "/home/user/test.txt"
    desktop.filesystem.write(file, "world!")
    
    # Normally, we would use `desktop.process.start_and_wait()` to run a new process
    # and wait until it finishes.
    # However, the mousepad command does not exit until you close the window so we
    # we need to just start the process and run it in the background so it doesn't
    # block our code.
    desktop.process.start(
        f"mousepad {file}",
        env_vars={"DISPLAY": desktop.DISPLAY},
        on_stderr=lambda stderr: print(stderr),
        on_stdout=lambda stdout: print(stdout),
        cwd="/home/user",
    )
    time.sleep(2)  
    #####
    
    desktop.screenshot("screenshot-2.png")

    # Write "Hello, " in the text editor
    desktop.pyautogui(
        """
pyautogui.write("Hello, ")
"""
    )
    desktop.screenshot("screenshot-3.png")
