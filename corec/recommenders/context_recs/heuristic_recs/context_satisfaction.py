import numpy as np
import pandas as pd
from pydantic import (
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PrivateAttr,
    PositiveInt,
    validate_arguments,
)

from typing import Union, Optional, List

from .heuristic_rec import HeuristicRec


# NOTE: Located out of the recommender class to import it from 'evaluation' module
def context_satisfaction(
    ctx_rec: np.ndarray,
    ctx_i_matrix: np.ndarray,
    alpha: NonNegativeFloat = 0,
):
    """
    Calculate the context satisfaction score for a set of recommendations based on the intersection
    and union of the recommended items with the target context.

    Args:
        `ctx_rec`: The recommended item context.
        `ctx_i_matrix`: A matrix where each row represents a user's query context.
        `alpha`: A penalty factor for the unfulfillment of the query context.

    Returns:
        `np.ndarray`: An array containing the satisfaction score for each user based on the context.

    Explanation:
        The satisfaction score is calculated using the formula:
        Satisfaction = (|Intersection| / |Union|) + (alpha * |Diff| / sum(ctx_rec))
    """
    intersect = np.sum((ctx_i_matrix == ctx_rec) & (ctx_i_matrix != 0), axis=1)
    union = np.sum((ctx_i_matrix | ctx_rec), axis=1)
    diff = np.sum((ctx_rec != 0) & (ctx_i_matrix == 0), axis=1)
    union = np.where(union == 0, 1, union)

    return intersect / (union + alpha * diff / np.sum(ctx_rec))


class ContextSatisfactionRec(HeuristicRec):
    """
    Context-aware recommender system that selects items based on context satisfaction.
    The recommendations aim to maximize the overlap between the recommended items
    and the target context, with a penalty factor to account for unfulfilled context requirements.
    """

    alpha: NonNegativeFloat = Field(
        default=0,
        ge=0,
        le=1,
        description="Factor in the range [0,1] that penalizes the unfulfillment of the target context.",
    )
    _data_ncols: NonNegativeInt = PrivateAttr()
    _item_ctx_df: pd.DataFrame = PrivateAttr()

    class Config:
        extra = "forbid"

    def model_post_init(self, __context):
        super().model_post_init(__context)

        self._data_ncols = self._data_df.shape[1]
        self._item_ctx_df = self._data_df.iloc[
            :, [1] + list(range(3, self._data_ncols))
        ].drop_duplicates()

    @validate_arguments
    def get_top_k(
        self,
        context: List[int],
        user_id: Optional[Union[str, int]] = None,
        K: Optional[PositiveInt] = None,
    ):
        """
        Retrieves the top-K items based on context satisfaction, which measures how well
        the recommendations match the provided context.

        Args:
            `context`: The query context to filter the data.
            `user_id`: The query user ID (actually not used).
            `K`: The number of top items to retrieve. If `None`, all distinct items from the dataset will be returned.

        Returns:
            `Tuple[np.ndarray, np.ndarray]`:
                - The first array contains the IDs of the top-K items based on context satisfaction.
                - The second array contains the corresponding context satisfaction scores.
        """
        ctx_rec = np.array(context).astype(int)
        cxt_i_matrix = self._item_ctx_df.iloc[:, 1 : self._data_ncols - 2].values
        self._item_ctx_df["sat"] = context_satisfaction(
            ctx_rec, cxt_i_matrix, self.alpha
        )
        num_preds = K if K is not None else len(self._item_ctx_df)
        top_k = self._item_ctx_df.nlargest(num_preds, "sat")

        return (
            top_k.iloc[:, 0].values,
            top_k["sat"].values,
        )
