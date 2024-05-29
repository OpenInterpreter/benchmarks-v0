import uuid
import logging

from typing import Any, Callable, Optional
from e2b import Sandbox, EnvVars, ProcessMessage
from e2b.constants import TIMEOUT

logger = logging.getLogger(__name__)


class Desktop(Sandbox):
    DISPLAY = ":99"  # Must be the same as the start script env var
    template = "desktop"

    def __init__(
        self,
        template: Optional[str] = None,
        api_key: Optional[str] = None,
        cwd: Optional[str] = None,
        env_vars: Optional[EnvVars] = None,
        timeout: Optional[float] = TIMEOUT,
        on_stdout: Optional[Callable[[ProcessMessage], Any]] = None,
        on_stderr: Optional[Callable[[ProcessMessage], Any]] = None,
        on_exit: Optional[Callable[[int], Any]] = None,
        **kwargs,
    ):
        super().__init__(
            template=template or self.template,
            api_key=api_key,
            cwd=cwd,
            env_vars=env_vars,
            timeout=timeout,
            on_stdout=on_stdout,
            on_stderr=on_stderr,
            on_exit=on_exit,
            **kwargs,
        )

    def screenshot(self, name: str):
        screenshot_path = f"/home/user/screenshot-{uuid.uuid4()}.png"

        logger.info("Capturing screenshot")
        self.process.start_and_wait(
            f"scrot --pointer {screenshot_path}",
            env_vars={"DISPLAY": self.DISPLAY},
            on_stderr=lambda stderr: logger.debug(stderr),
            on_stdout=lambda stdout: logger.debug(stdout),
            cwd="/home/user",
        )

        logger.info("Downloading screenshot")
        file = self.download_file(screenshot_path)
        with open(name, "wb") as f:
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
        self.filesystem.write(code_path, code)

        self.process.start_and_wait(
            f"python {code_path}",
            on_stdout=lambda stdout: logger.debug(stdout),
            on_stderr=lambda stderr: logger.debug(stderr),
            env_vars={"DISPLAY": self.DISPLAY},
        )
