from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import Field, NonNegativeInt, PositiveInt, PrivateAttr, validate_call

from ..base_rec import BaseRec


class HeuristicRec(BaseRec, ABC):
    """
    Parent class for context-based recommenders utilizing heuristics.

    Columns from the dataset that are not explicitly specified as user, item,
    or rating columns will be treated as context features for the recommendation process.
    """

    chunk_size: Optional[PositiveInt] = Field(
        default=None,
        description="Size of the chunks in which the test data will be split. If None, the entire test dataset will be used without splitting.",
    )
    max_num_workers: Optional[PositiveInt] = Field(
        default=None,
        description="Maximum number of processors to use for parallel processing during the computation of predictions. If None, all available processors from the machine will be used.",
    )
    _test_df: pd.DataFrame = PrivateAttr()
    _data_df: pd.DataFrame = PrivateAttr()
    _preds_cols_types: dict = PrivateAttr()

    def model_post_init(self, __context):
        super().model_post_init(__context)

        self._test_df = pd.read_csv(
            self.test_path, sep=self.dataset_sep, compression=self.dataset_compression
        )
        train_df = pd.read_csv(
            self.train_path, sep=self.dataset_sep, compression=self.dataset_compression
        )
        self._data_df = train_df

        if self.valid_path is not None:
            valid_df = pd.read_csv(
                self.valid_path,
                sep=self.dataset_sep,
                compression=self.dataset_compression,
            )
            self._data_df = pd.concat([self._data_df, valid_df])

        context_idxs = [
            idx
            for idx in range(self._data_df.shape[1])
            if idx
            not in [
                self.dataset_user_idx,
                self.dataset_item_idx,
                self.dataset_rating_idx,
            ]
        ]

        std_col_order = [
            self.dataset_user_idx,
            self.dataset_item_idx,
            self.dataset_rating_idx,
        ] + context_idxs
        self._data_df = self._data_df.iloc[:, std_col_order]

        self._preds_cols_types = {
            self.preds_user_col_name: self._test_df.dtypes.iloc[self.dataset_user_idx],
            self.preds_item_col_name: self._test_df.dtypes.iloc[self.dataset_item_idx],
            self.preds_score_col_name: float,
            self.preds_test_item_col_name: self._test_df.dtypes.iloc[
                self.dataset_item_idx
            ],
        }

    @abstractmethod
    def get_top_k(
        self,
        context: List[int],
        user_id: Optional[Union[str, int]] = None,
        K: Optional[PositiveInt] = None,
    ):
        pass

    @validate_call
    def compute_predictions(self, output_path: str, K: Optional[PositiveInt] = None):
        """
        Compute heuristic predictions for the test dataset and save the results to a specified output file.

        This method splits the test dataset into chunks, processes each chunk in parallel with the
        `get_top_k()` method from the class and combines and saves the results in the specified file.

        Args:
            `output_path`: The path where the predictions file will be saved. The directory will be created if it does not exist.
            `K`: The number of recommendations to compute for each query in the test dataset. If None, the maximum possible number of recommendations will be computed.
        """
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        test_matrix = self._test_df.to_numpy()

        self._logger.info("Splitting test data...")
        if self.chunk_size is None:
            num_chunks = 1
        else:
            num_rows = test_matrix.shape[0]
            num_chunks = num_rows // self.chunk_size + (
                1 if num_rows % self.chunk_size != 0 else 0
            )
        chunks_ids = np.arange(num_chunks)

        self._logger.info("Prediction process starts")
        results = None
        with ProcessPoolExecutor(max_workers=self.max_num_workers) as executor:
            worker = partial(self._process_chunk, test_matrix=test_matrix, K=K)
            results = list(executor.map(worker, chunks_ids))

        results = np.vstack(results)

        self._logger.info("Writing predictions to output file...")
        results_df = pd.DataFrame(results, columns=self._preds_cols_types.keys())
        results_df = results_df.astype(self._preds_cols_types)
        results_df.to_csv(
            output_path,
            sep=self.preds_sep,
            index=False,
            compression=self.preds_compression,
        )

        self._logger.info("Recommendation process completed")

    def _process_chunk(
        self,
        chunk_id: NonNegativeInt,
        test_matrix: np.ndarray,
        K: Optional[PositiveInt],
    ):
        self._logger.info(f"Processing chunk {chunk_id}...")

        if (chunk_id + 1) * self.chunk_size < test_matrix.shape[0]:
            chunk = test_matrix[
                chunk_id * self.chunk_size : (chunk_id + 1) * self.chunk_size
            ]
        else:
            chunk = test_matrix[chunk_id * self.chunk_size :]

        self._logger.info(
            "Executing from row {} to row {}...".format(
                chunk_id * self.chunk_size,
                (chunk_id + 1) * self.chunk_size
                if (chunk_id + 1) * self.chunk_size < test_matrix.shape[0]
                else test_matrix.shape[0],
            )
        )

        predictions = []
        last_percentage_logged = 0

        for i, row in enumerate(chunk):
            current_percentage = int(i / chunk.shape[0] * 100)

            if current_percentage >= last_percentage_logged + 10:
                self._logger.info(
                    f"Processed {current_percentage}% of chunk {chunk_id}"
                )
                last_percentage_logged = current_percentage

            user_id = row[0]
            test_item_id = row[1]
            context = row[3:]
            items, scores = self.get_top_k(context, user_id=user_id, K=K)

            result_matrix = np.zeros((items.shape[0], 4))
            result_matrix[:, 0] = user_id
            result_matrix[:, 1] = items
            result_matrix[:, 2] = scores
            result_matrix[:, 3] = test_item_id
            predictions.append(result_matrix)

        self._logger.info(f"Finished chunk {chunk_id}")

        return np.vstack(predictions)
