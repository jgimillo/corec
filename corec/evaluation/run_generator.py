import multiprocessing as mp
from functools import partial
from typing import List, Optional

import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
from pydantic import (
    BaseModel,
    Field,
    FilePath,
    NonNegativeInt,
    PositiveInt,
    validate_arguments,
)
from ranx import Run, fuse

from .constants import RANX_FUSE_METHODS, RANX_FUSE_NORMS
from .utils import (
    chunkify_df,
    group_to_dict,
    save_json,
)


class RunGenerator(BaseModel):
    """Class that generates ranx `Run` objects from already computed predictions."""

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
        default="gzip",
        description="Compression type used in the predictions file.",
    )
    num_processors: PositiveInt = Field(
        default=1,
        description="Number of processes to run in parallel.",
    )

    class Config:
        extra = "forbid"

    def _store_run(self, run_dict: dict, output_path: Optional[str] = None):
        """Optionally saves the Run dictionary to a JSON file and returns the loaded Run."""
        if output_path is not None:
            save_json(run_dict, output_path)

        return Run(run_dict)

    def _run_value_func(self, group: DataFrameGroupBy, K: Optional[PositiveInt] = None):
        """
        Transforms a grouped DataFrame into a dictionary where the key is
        the item id and the value is the score.
        """
        return {
            row.iloc[self.preds_item_idx]: float(row.iloc[self.preds_score_idx])
            for _, row in (group if K is None else group.head(K)).iterrows()
        }

    @validate_arguments
    def compute_run(
        self,
        predictions_path: FilePath,
        K: Optional[PositiveInt] = None,
        output_path: Optional[str] = None,
    ):
        """
        Constructs a Run by retrieving, for each user-test item pair from the
        predictions file, all associated predicted item-score values. Optionally,
        saves the result as a JSON file.

        Args:
            `predictions_path`: Path to the file containing the predictions.
            `K`: Number of top predictions to retain per user.  If not specified, all predictions will be considered.
            `output_path`: Path to save the Run dict as a JSON file. If not specified, the Run is not saved.

        Returns:
            `ranx.Run`: The Run object computed from the predictions.
        """
        preds_df = pd.read_csv(
            predictions_path, sep=self.preds_sep, compression=self.preds_compression
        )
        item_col_name = preds_df.columns[self.preds_item_idx]
        preds_df[item_col_name] = preds_df[item_col_name].astype(str)

        chunks = chunkify_df(preds_df, self.num_processors)
        group_keys = [
            preds_df.columns[self.preds_user_idx],
            preds_df.columns[self.preds_test_item_idx],
        ]
        process_chunk = partial(
            group_to_dict,
            group_keys=group_keys,
            value_func=partial(self._run_value_func, K=K),
        )

        with mp.Pool(processes=self.num_processors) as pool:
            results = pool.map(process_chunk, chunks)

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
        output_path: Optional[str] = None,
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
