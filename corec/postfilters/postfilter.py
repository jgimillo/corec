from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    Field,
    FilePath,
    NonNegativeInt,
    PrivateAttr,
    validate_arguments,
)

from ..utils import get_context_lookup_dict


class PostFilter(BaseModel):
    """Class to post-filter predictions based on training (and validation) data."""

    train_path: FilePath = Field(
        default=...,
        description="Train data file path. If not specified, context filtering will not be performed.",
    )
    dataset_ctx_idxs: Union[List[NonNegativeInt], NonNegativeInt] = Field(
        ...,
        description="Context column index(es) in the dataset.",
    )
    preds_user_idx: NonNegativeInt = Field(
        default=0,
        description="Index for the user id column in the predictions.",
    )
    preds_item_idx: NonNegativeInt = Field(
        default=1,
        description="Index for the recommended item id column in the predictions.",
    )
    preds_test_item_idx: NonNegativeInt = Field(
        default=3,
        description="Index for the query item id column in the predictions.",
    )
    preds_sep: str = Field(
        default="\t",
        description="Separator used in the predictions file.",
    )
    preds_compression: Optional[str] = Field(
        default="gzip",
        description="Compression type used in the predictions file.",
    )
    valid_path: Optional[FilePath] = Field(
        default=None,
        description="Validation data file path.",
    )
    dataset_user_idx: NonNegativeInt = Field(
        default=0,
        description="Index for the user id column in the dataset.",
    )
    dataset_item_idx: NonNegativeInt = Field(
        default=1,
        description="Index for the item id column in the dataset.",
    )
    dataset_sep: str = Field(
        default="\t",
        description="Separator used in the dataset files.",
    )
    dataset_compression: Optional[str] = Field(
        default=None,
        description="Compression type used in the dataset files.",
    )
    _context_lookup = PrivateAttr()

    class Config:
        extra = "forbid"

    def model_post_init(self, _):
        if isinstance(self.dataset_ctx_idxs, list) and not len(self.dataset_ctx_idxs):
            raise ValueError("Context indexes list cannot be empty.")

        self._context_lookup = get_context_lookup_dict(
            train_path=self.train_path,
            dataset_ctx_idxs=self.dataset_ctx_idxs,
            valid_path=self.valid_path,
            dataset_item_idx=self.dataset_item_idx,
            dataset_sep=self.dataset_sep,
            dataset_compression=self.dataset_compression,
        )

    @validate_arguments
    def postfilter(self, preds_path: FilePath, output_path: str):
        """
        Loads a predictions file, filters out rows where the predicted item and the
        query item do not share at least one context, and saves the filtered predictions
        keeping the same format as the original file.

        Args:
            `preds_path`: Path to the CSV file containing the predictions.
            `output_path`: Path to save the post-filtered predictions file.
        """
        preds_df = pd.read_csv(
            preds_path,
            sep=self.preds_sep,
            compression=self.preds_compression,
        )

        preds_items = preds_df.iloc[:, self.preds_item_idx].astype(str).values
        preds_test_items = preds_df.iloc[:, self.preds_test_item_idx].astype(str).values

        items_contexts = np.array(
            [self._context_lookup.get(item, []) for item in preds_items]
        )
        test_items_contexts = np.array(
            [self._context_lookup.get(item, []) for item in preds_test_items]
        )

        valid_rows = np.array(
            [
                any(ctx in test_item_contexts for ctx in item_contexts)
                for item_contexts, test_item_contexts in zip(
                    items_contexts, test_items_contexts
                )
            ]
        )
        filtered_preds_df = preds_df[valid_rows]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        filtered_preds_df.to_csv(
            output_path,
            index=False,
            sep=self.preds_sep,
            compression=self.preds_compression,
        )
