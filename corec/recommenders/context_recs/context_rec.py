import logging
from pathlib import Path
from typing import Optional

from pydantic import Field, PrivateAttr

from ..base_rec import BaseRec


class ContextRec(BaseRec):
    logs_path: Optional[str] = Field(
        default=None,
        description="Path to the log file where recommender logs will be saved. If None, the default logger will be used.",
    )
    _logger: logging.Logger = PrivateAttr()

    def model_post_init(self, __context):
        super().model_post_init(__context)

        if self.logs_path is None:
            self._logger = logging.getLogger(__name__)
            return

        logs_dir = Path(self.logs_path).parent
        logs_dir.mkdir(parents=True, exist_ok=True)

        self._logger = logging.getLogger(f"{__name__}.{self.logs_path}")
        self._logger.setLevel(logging.INFO)

        if not self._logger.hasHandlers():
            file_handler = logging.FileHandler(
                self.logs_path, mode="w+", encoding="utf-8"
            )
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s: %(levelname)-.1s %(message)s")
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)
