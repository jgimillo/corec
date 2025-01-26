import csv
import re
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    Field,
    FilePath,
    NonNegativeInt,
    PositiveInt,
    PrivateAttr,
)
from ranx import Qrels, Run, evaluate

from ..recommenders.context_recs.heuristic_recs.context_satisfaction import (
    context_satisfaction,
)
from .constants import (
    CUSTOM_METRICS,
    CUTOFF_RANX_METRICS,
    NON_CUTOFF_RANX_METRICS,
    RANX_METRICS,
)


class FuseRun(BaseModel):
    """Auxiliary class to group the fuse `Run` along with some of the fuse parameters."""

    run: Run = Field(
        ...,
        description="The computed fuse Run.",
    )
    fused_run_names: List[str] = Field(..., description="The names of the fused Runs.")
    norm: str = Field(..., description="The ranx norm used during the fusion.")
    method: str = Field(..., description="The ranx method used during the fusion.")

    class Config:
        arbitrary_types_allowed = True


class MetricGenerator(BaseModel):
    """
    Class for generating metric reports from precomputed `Run` objects.

    Supported metrics:
    - **Ranx metrics**: Metrics supported by the `ranx` library.
                        Refer to the official documentation: https://amenra.github.io/ranx/metrics/
    - **Custom metrics**: Available only for `Run` objects with keys in the format `u<user_id>_i<item_id>`:
        - `mean_ctx_sat`: Calculates the average context satisfaction between test and predicted items.
        - `acc_ctx_sat`: Calculates the accumulated context satisfaction between test and predicted items.
    """

    qrels: Qrels = Field(
        ...,
        description="Qrels used for the metrics computation.",
    )
    train_path: Optional[FilePath] = Field(
        default=None,
        description="Path to the training data. If specified, context metrics will be available for use.",
    )
    valid_path: Optional[FilePath] = Field(
        default=None,
        description="Path to the validation data.",
    )
    dataset_item_idx: NonNegativeInt = Field(
        default=1,
        description="Index for the item id column in the dataset.",
    )
    dataset_ctx_idxs: List[NonNegativeInt] = Field(
        default=[],
        description="Context column indexes in the dataset.",
    )
    dataset_sep: str = Field(
        default="\t",
        description="Separator used in the dataset files.",
    )
    dataset_compression: Optional[str] = Field(
        default=None,
        description="Compression type used in the dataset files.",
    )
    _context_lookup = PrivateAttr(default=None)

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"

    def model_post_init(self, _):
        if self.train_path is None:
            if not len(self.dataset_ctx_idxs):
                raise ValueError("Context indexes list cannot be empty.")
            return

        item_ctx_df = pd.read_csv(
            self.train_path,
            sep=self.dataset_sep,
            compression=self.dataset_compression,
        )

        if self.valid_path is not None:
            valid_df = pd.read_csv(
                self.valid_path,
                sep=self.dataset_sep,
                compression=self.dataset_compression,
            )
            item_ctx_df = pd.concat([item_ctx_df, valid_df])

        ctx_cols_names = [item_ctx_df.columns[i] for i in self.dataset_ctx_idxs]
        item_col_name = item_ctx_df.columns[self.dataset_item_idx]

        item_ctx_df[item_col_name] = item_ctx_df[item_col_name].astype(str)
        item_ctx_df = item_ctx_df[[item_col_name] + ctx_cols_names].drop_duplicates()

        self._context_lookup = dict(
            zip(
                item_ctx_df[item_col_name],
                item_ctx_df[ctx_cols_names].apply(tuple, axis=1),
            )
        )

    @staticmethod
    def _get_cutoff_metrics(metrics: List[str], cutoffs: List[PositiveInt] = []):
        ranx_metrics = [metric for metric in metrics if metric in RANX_METRICS]
        custom_metrics = [metric for metric in CUSTOM_METRICS]

        if not len(ranx_metrics + custom_metrics):
            raise Exception("No valid metric was provided.")

        if not len(cutoffs):
            return ranx_metrics, custom_metrics

        ranx_cutoff_metrics = [
            f"{metric}@{cutoff}"
            for metric in metrics
            if metric in CUTOFF_RANX_METRICS
            for cutoff in cutoffs
        ]
        ranx_cutoff_metrics += [
            metric for metric in metrics if metric in NON_CUTOFF_RANX_METRICS
        ]

        return ranx_cutoff_metrics, custom_metrics

    @staticmethod
    def _split_cutoff_metric(compose_metric: str):
        return (
            compose_metric.split("@")
            if "@" in compose_metric
            else (compose_metric, None)
        )

    def _compute_ctx_sat_metric(
        self,
        run: Run,
        aggregation_fn: Callable,
        cutoffs: List[PositiveInt] = [],
    ):
        run_dict = run.to_dict()

        metrics = [] if not cutoffs else {cutoff: [] for cutoff in cutoffs}

        for user_test_item, preds in run_dict.items():
            # NOTE: Potential error point if 'user_id' or 'item_id' contains letters and '_'
            match = re.match(r"u(?P<user_id>[^_]+)_i(?P<item_id>.+)", user_test_item)
            test_item_id = match.group("item_id")

            test_item_ctx = np.array(self._context_lookup.get(test_item_id, []))
            preds_cutoff = list(preds.keys())[: max(cutoffs) if cutoffs else len(preds)]
            pred_ctx_matrix = np.array(
                [self._context_lookup.get(item_id, []) for item_id in preds_cutoff]
            )

            # If a Run contains a value with no predictions, we omit it
            if not len(pred_ctx_matrix):
                continue

            ctx_satisfaction_scores = context_satisfaction(
                ctx_rec=test_item_ctx, ctx_i_matrix=pred_ctx_matrix
            )

            if not cutoffs:
                metrics.append(aggregation_fn(ctx_satisfaction_scores))
            else:
                for cutoff in cutoffs:
                    metrics[cutoff].append(
                        aggregation_fn(ctx_satisfaction_scores[:cutoff])
                    )

        return (
            {cutoff: np.mean(metrics[cutoff]) for cutoff in cutoffs}
            if cutoffs
            else aggregation_fn(metrics)
        )

    def _compute_metrics(
        self,
        output_path: FilePath,
        runs: List[Union[Run, FuseRun]],
        metrics: List[str],
        cutoffs: List[PositiveInt] = [],
        is_fuse: bool = False,
    ):
        if not len(runs):
            return

        ranx_cutoff_metrics, custom_metrics = self._get_cutoff_metrics(
            metrics, cutoffs=cutoffs
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        file = open(output_path, mode="w", newline="", encoding="utf-8")
        csv_writer = csv.writer(file, delimiter=";")

        if is_fuse:
            max_fuse_runs = max(len(fuse_run.fused_run_names) for fuse_run in runs)
            cols_names = [f"Model {i + 1}" for i in range(max_fuse_runs)]
            cols_names += ["Fuse norm", "Fuse method", "Metric", "Cutoff", "Score"]
        else:
            cols_names = ["Model", "Metric", "Cutoff", "Score"]

        csv_writer.writerow(cols_names)

        for run in runs:
            base_row = (
                list(run.fused_run_names)
                + [None] * (max_fuse_runs - len(run.fused_run_names))
                + [run.norm, run.method]
                if is_fuse
                else [run.name]
            )

            # Ranx metrics
            if ranx_cutoff_metrics:
                ranx_scores = evaluate(
                    qrels=self.qrels,
                    run=run.run if is_fuse else run,
                    metrics=ranx_cutoff_metrics,
                    make_comparable=True,
                )
                for cutoff_metric in ranx_cutoff_metrics:
                    metric_name, cutoff = self._split_cutoff_metric(cutoff_metric)
                    score = ranx_scores[cutoff_metric]
                    csv_writer.writerow(
                        base_row + [metric_name, cutoff, f"{score:.4f}"]
                    )

            # Custom metrics
            for metric in custom_metrics:
                aggregation_fn = np.mean if metric == "mean_ctx_sat" else np.sum
                scores = self._compute_ctx_sat_metric(
                    run.run if is_fuse else run, aggregation_fn, cutoffs=cutoffs
                )
                if not cutoffs:
                    csv_writer.writerow(base_row + [metric, None, f"{scores:.4f}"])
                else:
                    for cutoff in cutoffs:
                        csv_writer.writerow(
                            base_row + [metric, cutoff, f"{scores[cutoff]:.4f}"]
                        )

        file.close()

    def compute_non_fuse_runs_metrics(
        self,
        output_path: FilePath,
        runs: List[Run],
        metrics: List[str],
        cutoffs: List[PositiveInt] = [],
    ):
        """
        Computes metrics for the provided non fuse runs and saves the results in a CSV file.

        Args:
            `output_path`: File path where the computed metrics will be saved as a CSV file.
            `runs`: List of Runs for which the metrics will be computed.
            `metrics`: List of metric names to compute. Not supported ones will be omitted.
            `cutoffs`: List of cutoff values to apply to the metrics.

        Raise:
            `ValueError`: If no valid metric was provided.
        """
        self._compute_metrics(
            output_path=output_path,
            runs=runs,
            metrics=metrics,
            cutoffs=cutoffs,
            is_fuse=False,
        )

    def compute_fuse_runs_metrics(
        self,
        output_path: FilePath,
        fuse_runs: List[FuseRun],
        metrics: List[str],
        cutoffs: List[PositiveInt] = [],
    ):
        """
        Computes metrics for the provided non fuse runs and saves the results in a CSV file.

        Args:
            `output_path`: File path where the computed metrics will be saved as a CSV file.
            `qrels`: Qrels used for metric computation.
            `fuse_runs`: List of FuseRun diccionaries for which the metrics will be computed.
            `metrics`: List of metric names to compute. Not supported ones will be omitted.
            `cutoffs`: List of cutoff values to apply to the metrics.

        Raise:
            `ValueError`: If no valid metric was provided.
        """
        self._compute_metrics(
            output_path=output_path,
            runs=fuse_runs,
            metrics=metrics,
            cutoffs=cutoffs,
            is_fuse=True,
        )
