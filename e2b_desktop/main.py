import uuid
import logging
from e2b import Sandbox

logger = logging.getLogger(__name__)


class Desktop(Sandbox):
    default_template = "desktop"

    def screenshot(self, name: str):
        screenshot_path = f"/home/user/screenshot-{uuid.uuid4()}.png"

        logger.info("Capturing screenshot")
        self.commands.run(
            f"scrot --pointer {screenshot_path}",
            on_stderr=lambda stderr: logger.debug(stderr),
            on_stdout=lambda stdout: logger.debug(stdout),
            cwd="/home/user",
        )

        logger.info("Downloading screenshot")
        with open(name, "wb") as f:
            file = self.files.read(screenshot_path, format="bytes")
            f.write(file)

    @staticmethod
    def _wrap_pyautogui_code(code: str):
        return f"""
import pyautogui
import os
import Xlib.display

display = Xlib.display.Display(os.environ["DISPLAY"])
pyautogui._pyautogui_x11._display = display

{code}
exit(0)
"""

    def pyautogui(self, pyautogui_code: str):
        logger.info("Running pyautogui code")

        code_path = f"/home/user/code-{uuid.uuid4()}.py"

        code = self._wrap_pyautogui_code(pyautogui_code)

        logger.info("Writing code")
        self.files.write(code_path, code)

        self.commands.run(
            f"python {code_path}",
            on_stdout=lambda stdout: logger.debug(stdout),
            on_stderr=lambda stderr: logger.debug(stderr),
        )
