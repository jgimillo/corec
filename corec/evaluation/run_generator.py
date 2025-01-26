import multiprocessing as mp
from functools import partial
from typing import Any, List, Optional

import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
from pydantic import (
    BaseModel,
    Field,
    FilePath,
    NonNegativeInt,
    PositiveInt,
    PrivateAttr,
    validate_arguments,
)
from ranx import Run, fuse

from ..postfilters.postfilter import PostFilter
from .constants import RANX_FUSE_METHODS, RANX_FUSE_NORMS
from .utils import (
    chunkify_df,
    chunkify_list,
    group_to_dict,
    save_json,
)


class RunGenerator(BaseModel):
    """Class that generates ranx `Run` objects from already computed predictions."""

    test_path: FilePath = Field(
        ...,
        description="Path to the test data. It is used to compute the item-context list.",
    )
    context_idxs: List[NonNegativeInt] = Field(
        ...,
        description="Context column indexes in the dataset files.",
    )
    train_path: Optional[FilePath] = Field(
        default=None,
        description="Path to the training data, used during the post-filtering.",
    )
    valid_path: Optional[FilePath] = Field(
        default=None,
        description="Path to the validation data, used during the post-filtering.",
    )
    dataset_sep: str = Field(
        default="\t",
        description="Separator used in the dataset files.",
    )
    dataset_compression: Optional[str] = Field(
        default=None,
        description="Compression type used in the dataset files.",
    )
    preds_user_idx: NonNegativeInt = Field(
        default=0,
        description="Index for the user id column in the predictions.",
    )
    preds_item_idx: NonNegativeInt = Field(
        default=1,
        description="Index for the item id column in the predictions.",
    )
    preds_score_idx: NonNegativeInt = Field(
        default=2,
        description="Index for the score column in the predictions.",
    )
    preds_test_item_idx: NonNegativeInt = Field(
        default=3,
        description="Index for the test item id column in the predictions.",
    )
    preds_sep: str = Field(
        default="\t",
        description="Separator used in the predictions file.",
    )
    preds_compression: Optional[str] = Field(
        default=None,
        description="Compression type used in the predictions file.",
    )
    num_processors: PositiveInt = Field(
        default=1,
        description="Number of processes to run in parallel.",
    )
    _item_context_list: List[Any] = PrivateAttr()

    class Config:
        extra = "forbid"

    def model_post_init(self, _):
        test_df = pd.read_csv(
            self.test_path, sep=self.dataset_sep, compression=self.dataset_compression
        )
        test_df.iloc[:, self.preds_item_idx] = test_df.iloc[
            :, self.preds_item_idx
        ].astype(str)

        self._item_context_list = [
            [
                row.iloc[self.preds_item_idx],
                tuple(row.iloc[self.context_idxs]),
                test_df.loc[
                    test_df.iloc[:, self.preds_item_idx]
                    == row.iloc[self.preds_item_idx],
                    test_df.columns[self.preds_user_idx],
                ].unique(),
            ]
            for _, row in test_df.drop_duplicates(
                subset=test_df.columns[self.preds_item_idx]
            ).iterrows()
        ]

    @staticmethod
    def _sanitize_inf_from_run_dict(run_dict: dict):
        """
        Replaces "-inf" score values in a Run dictionary with a very low value.
        """
        for _, scores in run_dict.items():
            for key, value in scores.items():
                if value == -float("inf"):
                    scores[key] = -1e10

        return run_dict

    def _store_run(self, run_dict: dict, output_path: str = None):
        """
        Optionally saves the Run dictionary to a JSON file and sanitizes
        it before returning the loaded Run.
        """
        if output_path is not None:
            save_json(run_dict, output_path)
        run_dict = self._sanitize_inf_from_run_dict(run_dict)
        return Run(run_dict)

    def _contextual_value_func(self, group: DataFrameGroupBy):
        """
        Transforms a grouped DataFrame into a dictionary where the key is
        the item id and the value is the score.
        """
        return {
            row.iloc[self.preds_item_idx]: float(row.iloc[self.preds_score_idx])
            for _, row in group.iterrows()
        }

    @validate_arguments
    def compute_contextual_run(
        self, predictions_path: FilePath, output_path: str = None
    ):
        """
        Constructs a Run by retrieving, for each user-test item pair from the
        predictions file, all associated predicted item-score values. Optionally,
        saves the result as a JSON file.

        Args:
            `predictions_path`: Path to the file containing the predictions.
            `output_path`: Path to save the Run dict as a JSON file. If not specified, the Run is not saved.

        Returns:
            `ranx.Run`: The Run object computed from the predictions.
        """
        preds_df = pd.read_csv(
            predictions_path, sep=self.dataset_sep, compression=self.preds_compression
        )
        preds_df.iloc[:, self.preds_item_idx] = preds_df.iloc[
            :, self.preds_item_idx
        ].astype(str)

        chunks = chunkify_df(preds_df, self.num_processors)
        group_keys = [
            preds_df.columns[self.preds_user_idx],
            preds_df.columns[self.preds_test_item_idx],
        ]
        process_chunk = partial(
            group_to_dict,
            group_keys=group_keys,
            value_func=self._contextual_value_func,
        )

        with mp.Pool(processes=self.num_processors) as pool:
            results = pool.map(process_chunk, chunks)

        run_dict = {
            key: value
            for partial_dict in results
            for key, value in partial_dict.items()
        }

        return self._store_run(run_dict, output_path=output_path)

    def _process_item_context(self, pf: PostFilter, chunk: list, K: int):
        """
        Processes a chunk of item-contexts from the test data and generates a
        dictionary of user-item predictions with the corresponding scores.
        """
        partial_dict = {}

        for item_id, context, user_ids in chunk:
            preds_df = pf.postfilter(context=context, user_ids=user_ids, K=K)
            user_preds_col_name = preds_df.columns[self.preds_user_idx]
            preds_df.iloc[:, self.preds_item_idx] = preds_df.iloc[
                :, self.preds_item_idx
            ].astype(str)

            partial_dict.update(
                {
                    f"u{user_id}_i{item_id}": dict(
                        zip(
                            group.iloc[:, self.preds_item_idx],
                            group.iloc[:, self.preds_score_idx],
                        )
                    )
                    for user_id, group in preds_df.groupby(user_preds_col_name)
                }
            )
        return partial_dict

    @validate_arguments
    def compute_non_contextual_run(
        self,
        predictions_path: FilePath,
        context_postfilter: bool = False,
        K: PositiveInt = None,
        output_path: str = None,
    ):
        """
        Constructs a Run by processing non-contextual predictions and optionally applying post-filtering.
        For each user-item pair from the test data, it includes the top K recommendations, optionally
        considering the exact same context between recommended and test item. The result can be saved
        as a JSON file.

        Args:
            `predictions_path`: Path to the predictions file.
            `context_postfilter`: Boolean flag to apply context post-filtering.
            `K`: Number of top predictions to retain per user during post-filtering.
            `output_path`: Path to save the Run dictionary as a JSON file. If not specified, the Run is not saved.

        Raises:
            `ValueError`: If `context_postfilter` is enabled but `train_path` was not provided during initialization.
        """
        if context_postfilter and self.train_path is None:
            raise ValueError("Cannot perform context postfiltering without train data.")

        pf = PostFilter(
            predictions_path=predictions_path,
            preds_user_idx=self.preds_user_idx,
            preds_sep=self.preds_sep,
            preds_compression=self.preds_compression,
            train_path=self.train_path if context_postfilter else None,
            valid_path=self.valid_path if context_postfilter else None,
            user_idx=self.preds_user_idx,
            item_idx=self.preds_item_idx,
            context_idxs=self.context_idxs,
            dataset_sep=self.dataset_sep,
            dataset_compression=self.dataset_compression,
        )

        chunks = chunkify_list(self._item_context_list, self.num_processors)
        process_item_with_pf = partial(self._process_item_context, pf, K=K)

        with mp.Pool(processes=self.num_processors) as pool:
            results = pool.map(process_item_with_pf, chunks)

        run_dict = {
            key: value
            for partial_dict in results
            for key, value in partial_dict.items()
        }

        return self._store_run(run_dict, output_path=output_path)

    @staticmethod
    def compute_fuse_run(
        runs: List[Run],
        norm: str = "min-max",
        method: str = "wsum",
        output_path: str = None,
    ):
        """
        Computes a fused Run applying the specified norm and method.
        Optionally, saves the result as a JSON file.

        Args:
            `runs`: List of Run objects to be fused.
            `norm`: Norm to apply to the runs before fusing.
            `method`: Method use for fusing.
            `output_path`: Path to save the fused Run as a JSON file. If not specified, the Run is not saved.

        Raises:
            `ValueError`: If the provided `norm` or `method` are not supported by ranx.

        Returns:
            `ranx.Run`: The Run object containing the combined predictions.
        """
        if norm not in RANX_FUSE_NORMS:
            raise ValueError(
                f"Invalid fuse norm value. Choose from: {RANX_FUSE_NORMS}."
            )
        if method not in RANX_FUSE_METHODS:
            raise ValueError(
                f"Invalid fuse method value. Choose from: {RANX_FUSE_METHODS}."
            )

        combined_run = fuse(runs=runs, norm=norm, method=method)

        if output_path is not None:
            save_json(combined_run.to_dict(), output_path)
        return combined_run
