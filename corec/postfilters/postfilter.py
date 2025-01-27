from typing import List, Optional

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


class PostFilter(BaseModel):
    """Class to post-filter predictions based on training (and validation) data."""

    predictions_path: FilePath = Field(..., description="Predictions file path.")
    preds_user_idx: NonNegativeInt = Field(
        default=0, description="Index for the user id column in the predictions."
    )
    preds_sep: str = Field(
        default="\t", description="Separator used in the predictions file."
    )
    preds_compression: str = Field(
        default=None, description="Compression type used in the predictions file."
    )
    train_path: Optional[FilePath] = Field(
        default=None,
        description="Train data file path. If not specified, context filtering will not be performed.",
    )
    valid_path: Optional[FilePath] = Field(
        default=None, description="Validation data file path."
    )
    user_idx: NonNegativeInt = Field(
        default=0, description="Index for the user id column in the test data."
    )
    item_idx: NonNegativeInt = Field(
        default=1, description="Index for the item id column in the test data."
    )
    context_idxs: List[NonNegativeInt] = Field(
        ..., description="Context column indexes in the dataset."
    )
    dataset_sep: str = Field(
        default="\t", description="Separator used in the dataset files."
    )
    dataset_compression: Optional[str] = Field(
        default=None, description="Compression type used in the dataset files."
    )
    _preds_df = PrivateAttr()
    _user_preds_col_name = PrivateAttr()
    _items_ids = PrivateAttr()
    _user_ids = PrivateAttr()
    _num_preds = PrivateAttr()
    _context_lookup = PrivateAttr(default=None)

    class Config:
        extra = "forbid"

    def model_post_init(self, _):
        self._preds_df = pd.read_csv(
            self.predictions_path,
            sep=self.preds_sep,
            compression=self.preds_compression,
        )
        self._user_preds_col_name = self._preds_df.columns[self.preds_user_idx]
        self._item_ids = self._preds_df.iloc[:, self.item_idx].values
        self._user_ids = self._preds_df.iloc[:, self.user_idx].values
        self._num_preds = self._preds_df.shape[0]

        if self.train_path is None:
            return

        item_ctx_df = pd.read_csv(
            self.train_path,
            sep=self.dataset_sep,
            compression=self.dataset_compression,
        )
        if self.valid_path is not None:
            valid_df = pd.read_csv(self.valid_path, sep=self.dataset_sep)
            item_ctx_df = pd.concat([item_ctx_df, valid_df])

        ctx_cols_names = [item_ctx_df.columns[i] for i in self.context_idxs]
        item_col_name = item_ctx_df.columns[self.item_idx]

        context_lookup = (
            item_ctx_df.groupby(ctx_cols_names)[item_col_name].apply(set).to_dict()
        )
        self._context_lookup = {
            key if isinstance(key, tuple) else (key,): value
            for key, value in context_lookup.items()
        }

    @validate_arguments
    def postfilter(self, context: tuple = None, user_ids: list = None, K: int = None):
        """
        Applies post-filtering to the predictions based on context or user IDs
        (or both), and an optional top-K cutoff.

        Args:
            `context`: A tuple representing the context for filtering (if train data was provided during initialization).
            `user_ids`: A list of user IDs to filter.
            `K`: Number of top predictions per user to retain. If not specified, all predictions are included.

        Returns:
            `pandas.DataFrame`: A DataFrame containing the filtered predictions based on the specified criteria.
        """
        if context is not None and self._context_lookup is not None:
            filtered_item_ids = self._context_lookup.get(context)

            if not filtered_item_ids:
                return pd.DataFrame(columns=self._preds_df.columns)

            ctx_mask = np.isin(self._item_ids, list(filtered_item_ids))
        else:
            ctx_mask = np.ones(self._num_preds, dtype=bool)

        if user_ids is not None:
            user_mask = np.isin(self._user_ids, user_ids)
        else:
            user_ids = np.ones(self._num_preds, dtype=bool)

        filtered_df = self._preds_df[user_mask & ctx_mask]

        if K is not None:
            filtered_df = (
                filtered_df.groupby(self._user_preds_col_name)
                .head(K)
                .reset_index(drop=True)
            )

        return filtered_df
