from typing import List, Optional, Union

from pydantic import PositiveInt, validate_call

from .heuristic_rec import HeuristicRec


class ContextPopRec(HeuristicRec):
    """
    Context-aware recommender that returns the top-K recommendations for a query
    item based on popularity within a specific context. The most popular items are
    determined using the combined train and validation data and must fully satisfy
    the provided query context.
    """

    class Config:
        extra = "forbid"

    @validate_call
    def get_top_k(
        self,
        context: List[int],
        user_id: Optional[Union[str, int]] = None,
        K: Optional[PositiveInt] = None,
    ):
        """
        Retrieves the top-K most popular items within a specified context.

        Args:
            `context`: The query context to filter the data.
            `user_id`: The query user ID (actually not used).
            `K`: The number of top items to retrieve. If `None`, all distinct items within the context will be returned.

        Returns:
            `Tuple[np.ndarray, np.ndarray]`:
                - The first array contains the IDs of the top-K items.
                - The second array contains the corresponding counts of their popularity.
        """
        mask = (self._data_df.iloc[:, 3:] == context).all(axis=1)
        df_context = self._data_df[mask]

        df_context = (
            df_context.groupby(df_context.iloc[:, 1]).size().reset_index(name="count")
        )
        num_preds = K if K is not None else len(df_context)
        top_k = df_context.nlargest(num_preds, "count")

        return (
            top_k.iloc[:, 0].values,
            top_k["count"].values,
        )
