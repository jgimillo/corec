from typing import List, Optional

import pandas as pd
from pydantic import (
    BaseModel,
    Field,
    FilePath,
    NonNegativeInt,
    PositiveInt,
    PrivateAttr,
    validate_arguments,
)

from .metric_generator import FuseRun, MetricGenerator
from .qrels_generator import QrelsGenerator
from .run_generator import RunGenerator


class Evaluator(BaseModel):
    """
    Class that integrates the functionality of QrelsGenerator, RunGenerator, and
    MetricGenerator for streamlined evaluation.
    """

    test_path: str = Field(
        ...,
        description="Path to the test data.",
    )
    preds_path_template: str = Field(
        ...,
        description="Template for the prediction file path. All placeholders '{model}' will be dynamically replaced with the model name.",
        examples=["predictions/{model}/{model}_preds.txt"],
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
        default=None,
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
    _qrels = PrivateAttr(default_factory=dict)
    _contextual_runs = PrivateAttr(default_factory=dict)
    _pure_non_ctx_runs = PrivateAttr(default_factory=dict)
    _postfilter_runs = PrivateAttr(default_factory=dict)
    _fuse_runs = PrivateAttr(default_factory=dict)

    class Config:
        extra = "forbid"

    def model_post_init(self, _):
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
            test_path=self.test_path,
            train_path=self.train_path,
            valid_path=self.valid_path,
            context_idxs=self.dataset_ctx_idxs,
            dataset_sep=self.dataset_sep,
            dataset_compression=self.dataset_compression,
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

    @validate_arguments
    def get_computed_summary(self):
        """
        Returns a dictionary summarizing the progress of computations. The dictionary includes:

        - Whether Qrels have been computed.
        - The names of Runs generated for contextual, pure non-contextual, and post-filtered models.
        - The names of fused Runs computed from the above models.
        """
        return {
            "qrels": len(self._qrels) != 0,
            "runs": {
                "contextual": list(self._contextual_runs.keys()),
                "non-contextual": {
                    "pure": list(self._pure_non_ctx_runs.keys()),
                    "postfilter": list(self._postfilter_runs.keys()),
                },
                "fuse": [
                    {
                        "runs": fuse_run.fused_run_names,
                        "norm": fuse_run.norm,
                        "method": fuse_run.method,
                    }
                    for fuse_run in self._fuse_runs.values()
                ],
            },
        }

    @validate_arguments
    def compute_qrels(self, output_path: Optional[str] = None):
        """
        Computes the test data Qrels using the class rating threshold and
        optionally saves the Qrels dictionary in a JSON file.

        Args:
            `output_path`: `output_path`: Path to save the Qrels dictionary as a JSON file. If not specified, the Qrels is not saved.
        """
        self._qrels = self._qrels_gen.compute_qrels(
            rating_thr=self.rating_thr,
            output_path=output_path,
        )

        self._metric_gen = MetricGenerator(
            qrels=self._qrels,
            train_path=self.train_path,
            valid_path=self.valid_path,
            dataset_item_idx=self.dataset_item_idx,
            dataset_ctx_idxs=self.dataset_ctx_idxs,
            dataset_sep=self.dataset_sep,
            dataset_compression=self.dataset_compression,
        )

    @validate_arguments
    def compute_contextual_run(self, ctx_model_name: str, run_name: str = None):
        """
        Generates the Run for the specified context-aware model and, if `runs_path_template` was
        specified during initialization, saves the Run dictionary in the corresponding JSON file.

        Args:
            `ctx_model_name`: Name of context-aware model for which Run will be generated.
            `run_name`: Name assigned to the generated Run and to be displayed in metrics tables. If None, the model name will be used as the default value.
        """
        if run_name is None:
            run_name = ctx_model_name

        predictions_path = self._get_preds_path(ctx_model_name)
        output_path = self._get_run_output_path(run_name)

        run = self._run_gen.compute_contextual_run(
            predictions_path=predictions_path,
            output_path=output_path,
        )
        run.name = run_name
        self._contextual_runs[run.name] = run

    @validate_arguments
    def compute_pure_non_contextual_run(
        self,
        non_ctx_model_name: str,
        run_name: str = None,
        K: int = None,
    ):
        """
        Generates the Run for the specified non-context-aware model without post-filtering and, if
        `runs_path_template` was specified during initialization, saves the Runs dictionaries
        in the corresponding JSON file.

        Args:
            `non_ctx_model_name`: Name of non-context-aware model for which Run will be generated.
            `run_name`: Name assigned to the generated Run and to be displayed in metrics tables. If None, the model name will be used as the default value.
            `K`: Number of top predictions to retain per user. By default all recommendations will be considered.
        """
        if run_name is None:
            run_name = non_ctx_model_name

        predictions_path = self._get_preds_path(non_ctx_model_name)
        output_path = self._get_run_output_path(run_name)

        if run_name is None:
            run_name = non_ctx_model_name

        run = self._run_gen.compute_non_contextual_run(
            predictions_path=predictions_path,
            K=K,
            output_path=output_path,
        )
        run.name = run_name
        self._pure_non_ctx_runs[run.name] = run

    @validate_arguments
    def compute_postfilter_run(
        self,
        non_ctx_model_name: str,
        run_name: str,
        K: int = None,
    ):
        """
        Generates Runs for the specified non-context-aware models with post-filtering applied
        and, if `runs_path_template` was specified during initialization, saves the Runs dictionaries
        in the corresponding JSON file.

        Args:
            `non_ctx_models_names`: List of names of non-context-aware models for which Runs will be generated.
            `run_name`: Name assigned to the generated Run and to be displayed in metrics tables. If None, the model name will be used as the default value.
            `K`: Number of top predictions to retain per user. By default all recommendations will be considered.
        """
        if run_name is None:
            run_name = non_ctx_model_name

        predictions_path = self._get_preds_path(non_ctx_model_name)
        output_path = self._get_run_output_path(run_name)

        run = self._run_gen.compute_non_contextual_run(
            predictions_path=predictions_path,
            context_postfilter=True,
            K=K,
            output_path=output_path,
        )
        run.name = run_name
        self._postfilter_runs[run.name] = run

    @validate_arguments
    def compute_fuse_run(
        self,
        ctx_run_names: List[str] = [],
        pure_non_ctx_run_names: List[str] = [],
        postfilter_run_names: List[str] = [],
        norm: str = "min-max",
        method: str = "wsum",
        run_name: str = None,
    ):
        """
        Generates a fused Run by combining the specified computed Runs, and normalization and
        fusion methods. The computed Run is optionally saved to a JSON file.

        Args:
            `ctx_run_names`: List of names of computed contextual Runs to be fused.
            `pure_non_ctx_run_names`: List of names of computed pure non-contextual Runs to be fused.
            `postfilter_run_names`: List of names of computed post-filtered Runs to be fused.
            `norm`: Ranx normalization method to apply before fusion.
            `method`: Ranx fusion method to apply.
            `run_name`: Name assigned to the generated Run. If None, a concatenation between the model names, norm and method will be used as the default value.

        Raises:
            `RuntimeError`: If any specified Run was not previously computed.
        """
        runs_to_check = {
            "contextual": (ctx_run_names, self._contextual_runs),
            "pure non-contextual": (pure_non_ctx_run_names, self._pure_non_ctx_runs),
            "postfilter": (postfilter_run_names, self._postfilter_runs),
        }

        runs_to_fuse = []
        for run_type, (check_names, run_dict) in runs_to_check.items():
            for check_name in check_names:
                if check_name not in run_dict:
                    raise RuntimeError(
                        f"{check_name} Run of type '{run_type}' is not yet computed."
                    )
                runs_to_fuse.append(run_dict[check_name])

        if not len(runs_to_fuse):
            return

        fused_run_names = [run.name for run in runs_to_fuse]
        if run_name is None:
            run_name = "+".join(fused_run_names)
            run_name += f"_{norm}_{method}"

        output_path = self._get_run_output_path(run_name)

        run = self._run_gen.compute_fuse_run(
            runs=runs_to_fuse, norm=norm, method=method, output_path=output_path
        )
        run.name = run_name

        self._fuse_runs[run.name] = FuseRun(
            run=run,
            fused_run_names=fused_run_names,
            norm=norm,
            method=method,
        )

    @validate_arguments
    def compute_metrics(
        self,
        non_fuse_output_path: str,
        fuse_output_path: str,
        metrics: List[str] = [],
        cutoffs: List[PositiveInt] = [],
    ):
        """
        Compute and save metrics in the specified CSV files for all the computed Runs.

        Args:
            `non_fuse_output_path`: Path to the CSV file where metrics for non-fused runs will be saved.
            `fuse_output_path`: Path to the CSV file where metrics for fused runs will be saved.
            `metrics`: List of metrics to compute.
            `cutoffs`: List of cutoffs to consider when calculating the metrics.

        Raises:
            `RuntimeError`: If Qrels were not previously computed.
        """
        if self._metric_gen is None:
            raise RuntimeError("Qrels haven't beeen computer yet.")

        non_fuse_runs = (
            list(self._contextual_runs.values())
            + list(self._pure_non_ctx_runs.values())
            + list(self._postfilter_runs.values())
        )

        self._metric_gen.compute_non_fuse_runs_metrics(
            output_path=non_fuse_output_path,
            runs=non_fuse_runs,
            metrics=metrics,
            cutoffs=cutoffs,
        )

        fuse_runs = list(self._fuse_runs.values())

        self._metric_gen.compute_fuse_runs_metrics(
            output_path=fuse_output_path,
            fuse_runs=fuse_runs,
            metrics=metrics,
            cutoffs=cutoffs,
        )
