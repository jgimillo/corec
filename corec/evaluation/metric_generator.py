import csv
import re
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from pydantic import (
    BaseModel,
    Field,
    FilePath,
    NonNegativeInt,
    PositiveInt,
    PrivateAttr,
)
from ranx import Qrels, Run, evaluate

from ..utils import context_satisfaction, get_context_lookup_dict
from .constants import (
    CUSTOM_METRICS,
    CUTOFF_RANX_METRICS,
    NON_CUTOFF_RANX_METRICS,
    RANX_METRICS,
)


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
    output_path: str = Field(
        ...,
        description="File path where the computed metrics will be saved.",
    )
    output_sep: str = Field(
        default="\t",
        description="Separator used in the output CSV files.",
    )
    preclear_output: bool = Field(
        default=True,
        description="Whether to clear the output file before adding new metrics.",
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
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

        if self.preclear_output:
            with open(self.output_path, mode="w", newline="", encoding="utf-8") as file:
                csv_writer = csv.writer(file, delimiter=self.output_sep)
                cols_names = [
                    "Models",
                    "Fuse norm",
                    "Fuse method",
                    "Metric",
                    "Cutoff",
                    "Score",
                ]
                csv_writer.writerow(cols_names)

        if self.train_path is None:
            return

        self._context_lookup = get_context_lookup_dict(
            train_path=self.train_path,
            dataset_ctx_idxs=self.dataset_ctx_idxs,
            valid_path=self.valid_path,
            dataset_item_idx=self.dataset_item_idx,
            dataset_sep=self.dataset_sep,
            dataset_compression=self.dataset_compression,
        )

    @staticmethod
    def _get_cutoff_metrics(metrics: List[str], cutoffs: List[PositiveInt] = []):
        if not len(metrics):
            return [], []

        ranx_metrics = [metric for metric in metrics if metric in RANX_METRICS]
        custom_metrics = [metric for metric in metrics if metric in CUSTOM_METRICS]

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

    def _get_item_context(self, item_id: str):
        return next(iter(self._context_lookup.get(item_id, set())), set())

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

            test_item_ctx = np.array(self._get_item_context(test_item_id))
            preds_cutoff = list(preds.keys())[: max(cutoffs) if cutoffs else len(preds)]

            pred_ctx_matrix = np.array(
                [self._get_item_context(item_id) for item_id in preds_cutoff]
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
        run: Run,
        metrics: List[str],
        cutoffs: List[PositiveInt] = [],
        fuse_norm: Optional[str] = None,
        fuse_method: Optional[str] = None,
    ):
        ranx_cutoff_metrics, custom_metrics = self._get_cutoff_metrics(
            metrics, cutoffs=cutoffs
        )

        file = open(self.output_path, mode="a", newline="", encoding="utf-8")
        csv_writer = csv.writer(file, delimiter=self.output_sep)
        base_row = [run.name, fuse_norm, fuse_method]

        # Ranx metrics
        if ranx_cutoff_metrics:
            ranx_scores = evaluate(
                qrels=self.qrels,
                run=run,
                metrics=ranx_cutoff_metrics,
                make_comparable=True,
            )
            for cutoff_metric in ranx_cutoff_metrics:
                metric_name, cutoff = self._split_cutoff_metric(cutoff_metric)
                score = ranx_scores[cutoff_metric]
                csv_writer.writerow(base_row + [metric_name, cutoff, f"{score:.4f}"])

        # Custom metrics
        for metric in custom_metrics:
            aggregation_fn = np.mean if metric == "mean_ctx_sat" else np.sum
            scores = self._compute_ctx_sat_metric(run, aggregation_fn, cutoffs=cutoffs)
            if not cutoffs:
                csv_writer.writerow(base_row + [metric, None, f"{scores:.4f}"])
            else:
                for cutoff in cutoffs:
                    csv_writer.writerow(
                        base_row + [metric, cutoff, f"{scores[cutoff]:.4f}"]
                    )

        file.close()

    def compute_non_fuse_run_metrics(
        self,
        run: Run,
        metrics: List[str],
        cutoffs: List[PositiveInt] = [],
    ):
        """
        Computes metrics for the provided non fuse Run and saves the results in a CSV file.

        Args:
            `run`: Run for which the metrics will be computed.
            `metrics`: List of metric names to compute. Not supported ones will be omitted.
            `cutoffs`: List of cutoff values to apply to the metrics.

        Raise:
            `ValueError`: If no valid metric was provided.
        """
        self._compute_metrics(
            run=run,
            metrics=metrics,
            cutoffs=cutoffs,
        )

    def compute_fuse_run_metrics(
        self,
        fuse_run: Run,
        fuse_norm: str,
        fuse_method: str,
        metrics: List[str],
        cutoffs: List[PositiveInt] = [],
    ):
        """
        Computes metrics for the provided non fuse runs and saves the results in a CSV file.

        Args:
            `fuse_run`: Run for which the metrics will be computed.
            `fuse_norm`: Fuse norm to be indicated in the CSV file.
            `fuse_method`: Fuse method to be indicated in the CSV file.
            `metrics`: List of metric names to compute. Not supported ones will be omitted.
            `cutoffs`: List of cutoff values to apply to the metrics.

        Raise:
            `ValueError`: If no valid metric was provided.
        """
        self._compute_metrics(
            run=fuse_run,
            metrics=metrics,
            cutoffs=cutoffs,
            fuse_norm=fuse_norm,
            fuse_method=fuse_method,
        )
