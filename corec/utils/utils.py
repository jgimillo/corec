from typing import List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import FilePath, NonNegativeFloat, NonNegativeInt, validate_arguments


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


@validate_arguments
def get_context_lookup_dict(
    train_path: FilePath,
    dataset_ctx_idxs: Union[List[NonNegativeInt], NonNegativeInt],
    valid_path: Optional[FilePath] = None,
    dataset_item_idx: NonNegativeInt = 1,
    dataset_sep: str = "\t",
    dataset_compression: Optional[str] = None,
):
    """
    Processes the training and validation datasets to create a mapping
    where each item is linked to a set of contexts it appears in.

    Args:
        `train_path`: Path to the training dataset file.
        `dataset_ctx_idxs`: Column index(es) indicating the context information in the dataset.
        `valid_path`: Path to the validation dataset file.
        `dataset_item_idx`: Column index representing the item ID in the dataset files.
        `dataset_sep`: Delimiter used in the dataset files.
        `dataset_compression`: Compression format used for the dataset files.

    Raises:
        `ValueError`: If the context indexes list is empty.

    Returns:
        `dict`: A dictionary where keys are item IDs as strings and values are sets of associated contexts.
    """
    if not len(dataset_ctx_idxs):
        raise ValueError("Context indexes list cannot be empty.")

    if isinstance(dataset_ctx_idxs, int):
        dataset_ctx_idxs = [dataset_ctx_idxs]

    item_ctx_df = pd.read_csv(
        train_path,
        sep=dataset_sep,
        compression=dataset_compression,
    )
    if valid_path is not None:
        valid_df = pd.read_csv(
            valid_path,
            sep=dataset_sep,
            compression=dataset_compression,
        )
        item_ctx_df = pd.concat([item_ctx_df, valid_df])

    ctx_cols_names = [item_ctx_df.columns[i] for i in dataset_ctx_idxs]
    item_col_name = item_ctx_df.columns[dataset_item_idx]

    context_lookup = (
        item_ctx_df.groupby(item_col_name)[ctx_cols_names]
        .apply(lambda x: set(tuple(i) if len(i) > 1 else i[0] for i in x.values))
        .to_dict()
    )

    return {str(k): v for k, v in context_lookup.items()}
