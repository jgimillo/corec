from typing import List, Union, Optional

import numpy as np
from pydantic import PositiveInt, validate_arguments

from .heuristic_rec import HeuristicRec


class ContextRandomRec(HeuristicRec):
    """
    Context-aware recommender system that selects random items from the dataset
    within a given context. The recommendations are based on the available items
    for that specific context, with random shuffling to ensure diversity in the suggestions.
    """

    class Config:
        extra = "forbid"

    @validate_arguments
    def get_top_k(
        self,
        context: List[int],
        user_id: Optional[Union[str, int]] = None,
        K: Optional[PositiveInt] = None,
    ):
        """
        Retrieves the top-K random items within a specified context.

        Args:
            `context`: The query context to filter the data.
            `user_id`: The query user ID (actually not used).
            `K`: The number of top items to retrieve. If `None`, all distinct items within the context will be returned.

        Returns:
            `Tuple[np.ndarray, np.ndarray]`:
                - The first array contains the IDs of the randomly selected top-K items.
                - The second array contains the corresponding scores, which are assigned in descending order.
        """
        mask = (self._data_df.iloc[:, 3:] == context).all(axis=1)
        df_context = self._data_df[mask]

        distinct_items = np.unique(df_context.iloc[:, 1].values)
        N = min(len(distinct_items), K) if K is not None else len(distinct_items)

        np.random.shuffle(distinct_items)
        scores = np.arange(N, 0, -1)

        return distinct_items[:N], scores[:N]
