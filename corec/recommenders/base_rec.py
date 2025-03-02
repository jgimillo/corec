import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field, FilePath, NonNegativeInt, PrivateAttr


class BaseRec(BaseModel):
    """Base class for the `corec` recommender module."""

    train_path: Optional[FilePath] = Field(
        ...,
        description="Path to the training data.",
    )
    test_path: str = Field(
        ...,
        description="Path to the test data.",
    )
    valid_path: Optional[FilePath] = Field(
        default=None,
        description="Path to the validation data.",
    )
    dataset_user_idx: NonNegativeInt = Field(
        default=0,
        description="Index for the user id column in the dataset.",
    )
    dataset_item_idx: NonNegativeInt = Field(
        default=1,
        description="Index for the item id column in the dataset.",
    )
    dataset_rating_idx: NonNegativeInt = Field(
        default=2,
        description="Index for the rating column in the dataset.",
    )
    dataset_sep: str = Field(
        default="\t",
        description="Separator used in the dataset files.",
    )
    dataset_compression: Optional[str] = Field(
        default=None,
        description="Compression type used in the dataset files.",
    )
    preds_user_col_name: Optional[str] = Field(
        default=None,
        description="Column name used for the user ID in the predictions file. If None, the user column name from the dataset will be used.",
    )
    preds_item_col_name: Optional[str] = Field(
        default=None,
        description="Column name used for the predicted item ID. If None, the item column name from the dataset will be used.",
    )
    preds_score_col_name: str = Field(
        default="score:float",
        description="Column name used for the predicted score.",
    )
    preds_test_item_col_name: Optional[str] = Field(
        default=None,
        description="Column name used for the query item ID in the predictions file. If None, the item column name from the dataset will be used, prefixed with 'test_'.",
    )
    preds_sep: str = Field(
        default="\t",
        description="Separator used in the predictions files.",
    )
    preds_compression: Optional[str] = Field(
        default="gzip",
        description="Compression type used in the predictions files.",
    )
    logs_path: Optional[str] = Field(
        default=None,
        description="Path to the log file where recommender logs will be saved. If None, the default logger will be used.",
    )
    _logger: logging.Logger = PrivateAttr()

    def model_post_init(self, __context):
        super().model_post_init(__context)

        # Logger setup
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

        # Dataset features setup
        if not self.preds_user_col_name or not self.preds_item_col_name:
            test_df = pd.read_csv(
                self.test_path,
                sep=self.dataset_sep,
                compression=self.dataset_compression,
            )

            self.preds_user_col_name = (
                self.preds_user_col_name or test_df.columns[self.dataset_user_idx]
            )
            self.preds_item_col_name = (
                self.preds_item_col_name or test_df.columns[self.dataset_item_idx]
            )

        self.preds_test_item_col_name = (
            self.preds_test_item_col_name or f"test_{self.preds_item_col_name}"
        )
