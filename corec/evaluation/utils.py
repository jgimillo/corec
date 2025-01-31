import json
from pathlib import Path
from typing import Callable, Dict, List

from pandas import DataFrame
from pydantic import PositiveInt


def save_json(data: Dict, output_path: str):
    """Saves a dictionary to a JSON file, creating parent directories if necessary."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)


def group_to_dict(df: DataFrame, group_keys: List[str], value_func: Callable):
    """
    Groups a DataFrame by the specified keys and transforms each group into a dictionary.
    - group_keys: Columns to group by (used as the dictionary key).
    - value_func: A function that transforms each group into the dictionary value.
    """
    grouped_dict = {}
    for keys, group in df.groupby(group_keys):
        values = value_func(group)
        if len(values):
            key = "_".join(f"{prefix}{value}" for prefix, value in zip("ui", keys))
            grouped_dict[key] = values
    return grouped_dict


def chunkify_list(lst: list, n: PositiveInt):
    """Splits a list into n nearly equal-sized chunks."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def chunkify_df(df: DataFrame, n: PositiveInt):
    """Splits a DataFrame into n nearly equal-sized chunks."""
    k, m = divmod(len(df), n)
    return [df.iloc[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]
