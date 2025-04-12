import os
from typing import List, Optional

import pandas as pd
from pydantic import (
    BaseModel,
    Field,
    FilePath,
    NonNegativeInt,
    PositiveInt,
    PrivateAttr,
    validate_call,
)
from ranx import Run

from .metric_generator import MetricGenerator
from .qrels_generator import QrelsGenerator
from .run_generator import RunGenerator


class Evaluator(BaseModel):
    """
    Class that integrates the functionality of QrelsGenerator, RunGenerator, and
    MetricGenerator for streamlined evaluation.
    """

    test_path: FilePath = Field(
        ...,
        description="Path to the test data.",
    )
    preds_path_template: str = Field(
        ...,
        description="Template for the prediction file path. All placeholders '{model}' will be dynamically replaced with the model name.",
        examples=["predictions/{model}/{model}_preds.txt"],
    )
    metrics: List[str] = Field(
        ...,
        description="List of metrics to compute.",
    )
    cutoffs: List[PositiveInt] = Field(
        default=[],
        description="List of cutoffs to consider when calculating the metrics.",
    )
    output_path: Optional[str] = Field(
        default=None,
        description="Path to the CSV file where the computed metrics will be saved.",
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
        description="Path to the training data.",
    )
    valid_path: Optional[FilePath] = Field(
        default=None,
        description="Path to the validation data.",
    )
    dataset_user_idx: NonNegativeInt = Field(
        default=0,
        description="Index for the user id column in the dataset.",
    )
    dataset_item_idx: NonNegativeInt = Field(
        default=1,
        description="Index for the item id column in the dataset.",
    )
    dataset_rating_idx: NonNegativeInt = Field(
        default=2,
        description="Index for the rating column in the dataset.",
    )
    dataset_ctx_idxs: Optional[List[NonNegativeInt]] = Field(
        default=None,
        description="Context column indexes in the dataset. If None, all columns except user, item, and rating are used.",
    )
    dataset_sep: str = Field(
        default="\t",
        description="Separator used in the dataset files.",
    )
    dataset_compression: Optional[str] = Field(
        default=None,
        description="Compression type used in the dataset files.",
    )
    preds_user_idx: NonNegativeInt = Field(
        default=0,
        description="Index for the user id column in the predictions.",
    )
    preds_item_idx: NonNegativeInt = Field(
        default=1,
        description="Index for the item id column in the predictions.",
    )
    preds_score_idx: NonNegativeInt = Field(
        default=2,
        description="Index for the score column in the predictions.",
    )
    preds_test_item_idx: NonNegativeInt = Field(
        default=3,
        description="Index for the test item id column in the predictions.",
    )
    preds_sep: str = Field(
        default="\t",
        description="Separator used in the predictions file.",
    )
    preds_compression: Optional[str] = Field(
        default="gzip",
        description="Compression type used in the predictions file.",
    )
    runs_path_template: Optional[str] = Field(
        default=None,
        description=(
            "Template for the Runs output path. All placeholders '{run}' will be dynamically replaced with the run name. "
            "If not specified, runs dictionaries will not be saved."
        ),
        examples=["evaluation/{run}_run.json"],
    )
    rating_thr: NonNegativeInt = Field(
        int=0,
        description="Rating threshold for determining relevance in Qrels.",
    )
    num_processors: PositiveInt = Field(
        default=1,
        description="Number of processes to run in parallel.",
    )
    _qrels_gen = PrivateAttr(default=None)
    _run_gen = PrivateAttr()
    _metric_gen = PrivateAttr(default=None)

    class Config:
        extra = "forbid"

    def model_post_init(self, _):
        if not len(self.metrics):
            raise ValueError("You must provide at least one metric.")

        self._qrels_gen = QrelsGenerator(
            test_path=self.test_path,
            user_idx=self.dataset_user_idx,
            item_idx=self.dataset_item_idx,
            rating_idx=self.dataset_rating_idx,
            data_sep=self.dataset_sep,
            data_compression=self.dataset_compression,
        )

        if self.dataset_ctx_idxs is None:
            dataset_ncols = pd.read_csv(
                self.test_path,
                sep=self.dataset_sep,
                compression=self.dataset_compression,
            ).shape[1]

            excluded_idxs = {
                self.dataset_user_idx,
                self.dataset_item_idx,
                self.dataset_rating_idx,
            }

            self.dataset_ctx_idxs = [
                idx for idx in range(dataset_ncols) if idx not in excluded_idxs
            ]

        self._run_gen = RunGenerator(
            preds_user_idx=self.preds_user_idx,
            preds_item_idx=self.preds_item_idx,
            preds_score_idx=self.preds_score_idx,
            preds_test_item_idx=self.preds_test_item_idx,
            preds_sep=self.preds_sep,
            preds_compression=self.preds_compression,
            num_processors=self.num_processors,
        )

    def _get_preds_path(self, model_name: str):
        return self.preds_path_template.replace("{model}", model_name)

    def _get_run_output_path(self, run_name: str):
        if self.runs_path_template is None:
            return None

        return self.runs_path_template.replace("{run}", run_name)

    @validate_call
    def compute_qrels(self, output_path: Optional[str] = None):
        """
        Computes the test data Qrels using the class rating threshold and
        optionally saves the Qrels dictionary in a JSON file.

        Args:
            `output_path`: `output_path`: Path to save the Qrels dictionary as a JSON file. If not specified, the Qrels is not saved.
        """
        qrels = self._qrels_gen.compute_qrels(
            rating_thr=self.rating_thr,
            output_path=output_path,
        )

        self._metric_gen = MetricGenerator(
            qrels=qrels,
            output_path=self.output_path,
            output_sep=self.output_sep,
            preclear_output=self.preclear_output,
            train_path=self.train_path,
            valid_path=self.valid_path,
            dataset_item_idx=self.dataset_item_idx,
            dataset_ctx_idxs=self.dataset_ctx_idxs,
            dataset_sep=self.dataset_sep,
            dataset_compression=self.dataset_compression,
        )

    @validate_call
    def compute_run_metrics(
        self,
        model_name: str,
        run_name: Optional[str] = None,
        K: Optional[PositiveInt] = None,
        only_compute_run: bool = False,
    ):
        """
        Generates the Run for the specified recommender and, if `runs_path_template` was
        specified during initialization, saves the Run dictionary in the corresponding JSON file.
        Finally, the corresponding metrics are computed and stored in the output path specified
        during the initialization of the class.

        Args:
            `model_name`: Name of model for which Run will be generated. Its predictions will be loaded from the class attribute `preds_path_template`.
            `run_name`: Name assigned to the generated Run and to be displayed in metrics tables. If None, the model name will be used as the default value.
            `K`: Number of top predictions to retain per user. By default all recommendations will be considered.
            `only_compute_run`: Whether to only compute the Run and skip metrics calculation.

        Raises:
            `RuntimeError`: If Qrels were not previously computed.
        """
        if self._metric_gen is None:
            raise RuntimeError("Qrels haven't been computed yet.")

        if run_name is None:
            run_name = model_name

        predictions_path = self._get_preds_path(model_name)
        output_path = self._get_run_output_path(run_name)

        if output_path is None and only_compute_run:
            return

        run = self._run_gen.compute_run(
            predictions_path=predictions_path,
            K=K,
            output_path=output_path,
        )
        run.name = run_name

        if not only_compute_run:
            self._metric_gen.compute_non_fuse_run_metrics(
                run=run,
                metrics=self.metrics,
                cutoffs=self.cutoffs,
            )

    @validate_call
    def compute_fuse_metrics(
        self,
        run_names: List[str] = [],
        model_names: List[str] = [],
        norm: str = "min-max",
        method: str = "wsum",
        run_name: Optional[str] = None,
        only_compute_run: bool = False,
    ):
        """
        Generates a fused Run by combining the specified computed Runs using the given normalization
        and fusion methods. If any model name is provided, their Runs will be precomputed as well.
        The computed Run can optionally be saved to a JSON file.

        Args:
            `run_names`: List of names of computed Runs to be fused. They will be loaded from the class attribute `runs_path_template`.
            `model_names`: List of model names to compute the Run and include in the fusion process.
            `norm`: Ranx normalization method to apply before fusion.
            `method`: Ranx fusion method to apply.
            `run_name`: Name assigned to the generated Run. If None, a concatenation of the model names will be used as the default value.
            `only_compute_run`: Whether to only compute the Run and skip metrics calculation.

        Raises:
            RuntimeError: If Qrels or any specified Run was not previously computed.
        """
        if self._metric_gen is None:
            raise RuntimeError("Qrels haven't been computed yet.")

        runs_to_fuse = []

        for check_name in run_names:
            run_path = self._get_run_output_path(check_name)

            if not os.path.exists(run_path):
                raise RuntimeError(
                    f"'{check_name}' Run has not been found in the path '{run_path}'."
                )
            run = Run.from_file(run_path, name=check_name)
            runs_to_fuse.append(run)

        for model_name in model_names:
            predictions_path = self._get_preds_path(model_name)
            output_path = self._get_run_output_path(model_name)

            run = self._run_gen.compute_run(
                predictions_path=predictions_path,
                output_path=output_path,
            )
            run.name = model_name
            runs_to_fuse.append(run)

        if not len(runs_to_fuse):
            return

        fused_run_names = [run.name for run in runs_to_fuse]
        if run_name is None:
            run_name = " + ".join(fused_run_names)

        output_path = self._get_run_output_path(run_name)

        if output_path is None and only_compute_run:
            return

        fuse_run = self._run_gen.compute_fuse_run(
            runs=runs_to_fuse, norm=norm, method=method, output_path=output_path
        )
        fuse_run.name = run_name

        if not only_compute_run:
            self._metric_gen.compute_fuse_run_metrics(
                fuse_run=fuse_run,
                fuse_norm=norm,
                fuse_method=method,
                metrics=self.metrics,
                cutoffs=self.cutoffs,
            )
