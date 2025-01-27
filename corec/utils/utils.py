import numpy as np
from pydantic import NonNegativeFloat


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
