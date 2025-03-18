from typing import Optional

import pandas as pd
from pydantic import (
    BaseModel,
    Field,
    FilePath,
    NonNegativeInt,
    PrivateAttr,
    validate_call,
)
from ranx import Qrels

from .utils import group_to_dict, save_json


class QrelsGenerator(BaseModel):
    """Class that generates ranx `Qrels` from a test data file."""

    test_path: FilePath = Field(
        ...,
        description="Path to the test data.",
    )
    user_idx: NonNegativeInt = Field(
        default=0,
        description="Index for the user id column in the test data.",
    )
    item_idx: NonNegativeInt = Field(
        default=1,
        description="Index for the item id column in the test data.",
    )
    rating_idx: NonNegativeInt = Field(
        default=2,
        description="Index for the rating column in the test data.",
    )
    data_sep: str = Field(
        default="\t",
        description="Separator used in the test data file.",
    )
    data_compression: Optional[str] = Field(
        default=None,
        description="Compression type used in the test data file.",
    )
    _test_df = PrivateAttr()

    def model_post_init(self, _):
        self._test_df = pd.read_csv(
            self.test_path, sep=self.data_sep, compression=self.data_compression
        )
        item_col_name = self._test_df.columns[self.item_idx]
        self._test_df[item_col_name] = self._test_df[item_col_name].astype(str)

    class Config:
        extra = "forbid"

    @validate_call
    def compute_qrels(
        self, rating_thr: NonNegativeInt, output_path: Optional[str] = None
    ):
        """
        Computes Qrels from the test data and optionally saves the result as a JSON file.

        Args:
            `rating_thr`: Threshold for determining relevance. Ratings greater than or equal to this value are marked as relevant.
            `output_path`: Path to save the Qrels dictionary as a JSON file. If not specified, the Qrels is not saved.

        Returns:
            `ranx.Qrels`: The Qrels object containing the relevance judgments from the test data.
        """
        self._test_df["relevance"] = (
            self._test_df.iloc[:, self.rating_idx] >= rating_thr
        )
        self._test_df["relevance"] = self._test_df["relevance"].astype(int)

        qrels_dict = group_to_dict(
            df=self._test_df,
            group_keys=[
                self._test_df.columns[self.user_idx],
                self._test_df.columns[self.item_idx],
            ],
            value_func=lambda group: {
                group.iloc[0, self.item_idx]: int(row["relevance"])
                for _, row in group.iterrows()
            },
        )

        if output_path is not None:
            save_json(qrels_dict, output_path)
        return Qrels(qrels_dict)
