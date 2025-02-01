import gzip
import os
import re
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yaml
from elliot.run import run_experiment
from pydantic import (
    Field,
    FilePath,
    PositiveInt,
    PrivateAttr,
    validate_arguments,
)

from ..base_rec import BaseRec
from .constants import META_TEMPLATE, YAML_TEMPLATE


class ElliotRec(BaseRec):
    """
    A class that defines the configuration and processes for running an Elliot-based recommender system experiment.
    Since dataset files might contain context columns, they are unified in temporary files during the
    initialization of the class.
    """

    preds_path_template: str = Field(
        ...,
        description=(
            "Template for the predictions output path. All placeholders '{model}' will be dynamically replaced with the model name."
        ),
        examples=["predictions/{model}/{model}.f0.tsv"],
    )
    elliot_work_dir: str = Field(
        default=None,
        description="Directory used by Elliot for internal purposes, such as storing predictions, metrics, and weights. If None, a temporary directory will be created.",
    )
    _temp_train_path = PrivateAttr()
    _temp_test_path = PrivateAttr()
    _temp_valid_path = PrivateAttr()

    class Config:
        extra = "forbid"

    def model_post_init(self, __context):
        super().model_post_init(__context)

        self._temp_train_path = self._prepare_temp_dataset_file(self.train_path)
        self._temp_test_path = self._prepare_temp_dataset_file(self.test_path)
        self._temp_valid_path = self._prepare_temp_dataset_file(self.valid_path)

        self.elliot_work_dir = self.elliot_work_dir or tempfile.mkdtemp()

    def _get_train_items_count(self):
        train_df = pd.read_csv(
            self.train_path, sep=self.dataset_sep, compression=self.dataset_compression
        )
        return train_df.iloc[:, self.dataset_item_idx].nunique()

    def _prepare_temp_config_file(
        self, models_config: Dict[str, Dict], K: Optional[PositiveInt] = None
    ):
        """
        Prepares a temporary configuration file for the Elliot experiment.

        Args:
            `models_config`: Configuration for each model to be used in the experiment.
            `K`: The number recommendations to predict. If None, the number of distinct items in the train data will be used.

        Returns:
            `Tuple`: A tuple containing the path to the temporary configuration file and the path for the predictions output directory.
        """

        config = deepcopy(YAML_TEMPLATE)
        output_dir = Path(self.elliot_work_dir)

        config["data_config"]["train_path"] = self._temp_train_path
        config["data_config"]["test_path"] = self._temp_test_path
        if self._temp_valid_path is not None:
            config["data_config"]["valid_path"] = self._temp_valid_path

        config["path_log_folder"] = str(output_dir / "logs")
        config["path_output_rec_performance"] = str(output_dir / "performance")
        config["path_output_rec_result"] = str(output_dir / "predictions")
        config["path_output_rec_weight"] = str(output_dir / "weights")
        config["top_k"] = K if K is not None else self._get_train_items_count()

        for model_name, params in models_config.items():
            params.update(META_TEMPLATE)
            config["models"][model_name] = params

        with tempfile.NamedTemporaryFile(delete=False, mode="w") as temp_file:
            yaml.safe_dump({"experiment": config}, temp_file)

        return temp_file.name, config["path_output_rec_result"]

    def _prepare_temp_dataset_file(self, dataset_file_path: Optional[FilePath] = None):
        """
        Prepares temporary file by removing context columns from the specified dataset file.

        Args:
            `file_path`: Path to the dataset file to be processed.

        Returns:
            `str`: The path to the temporary dataset file.
        """
        if dataset_file_path is None:
            return None

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            df = pd.read_csv(
                dataset_file_path,
                sep=self.dataset_sep,
                compression=self.dataset_compression,
            )

            target_cols_idxs = [
                self.dataset_user_idx,
                self.dataset_item_idx,
                self.dataset_rating_idx,
            ]
            df = df.iloc[:, target_cols_idxs]
            df.to_csv(temp_file.name, sep=self.dataset_sep, index=False, header=False)

        return temp_file.name

    def _unify_predictions_files(
        self,
        messy_folder_path: str,
    ):
        """
        Organizes prediction files from a folder by model, selects the latest file for each model,
        adds a header, compresses it into gzip format, and saves it to a specified destination folder.
        Older files are removed.

        Args:
            `messy_folder_path`: Path to the folder containing the Elliot prediction files.
        """
        folder_path = Path(messy_folder_path).resolve()
        files = [file for file in folder_path.iterdir() if file.is_file()]

        model_files = {}
        pattern = r"^(?P<model>\w+)_"

        for file in files:
            match = re.match(pattern, file.name)
            model = match.group("model") if match else file.name.rsplit(".", 1)[0]

            if model not in model_files:
                model_files[model] = []
            model_files[model].append(file)

        for model, files in model_files.items():
            files.sort(key=lambda f: f.stat().st_ctime, reverse=True)
            last_file = files[0]
            feat_names = [
                self.preds_user_col_name,
                self.preds_item_col_name,
                self.preds_score_col_name,
            ]

            model_file = Path(self.preds_path_template.replace("{model}", model))
            model_file.parent.mkdir(parents=True, exist_ok=True)

            with open(last_file, "rt+") as f_in:
                content = f_in.read()
                f_in.seek(0, 0)
                f_in.write("\t".join(feat_names) + "\n" + content)

            if self.compress_preds:
                with open(last_file, "rb") as f_in:
                    with gzip.open(model_file, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                shutil.copy(last_file, model_file)

            for file in files:
                file.unlink()

    @validate_arguments
    def run_elliot_experiment(
        self,
        models_config: Dict[str, Dict],
        K: Optional[PositiveInt] = None,
        clean_elliot_work_dir: bool = False,
        clean_temp_dataset_files: bool = False,
    ):
        """
        Runs an Elliot experiment based on the specified parameters.

        This method prepares the temporary configuration file, runs the experiment, and then unifies
        the predictions files. It also removes temporary files after the experiment.

        Args:
            models_config: Configuration for the models used in the experiment (https://elliot.readthedocs.io/en/latest/guide/recommenders.html).
            K: Number of recommendations to predict. If None, the number of distinct items in the train data will be used.
            clean_elliot_work_dir: Whether to remove the Elliot work directory after the experiment.
            clean_temp_dataset_files: Whether to remove the temporary dataset files after the experiment.
        """
        try:
            temp_config_file, messy_preds_dir = self._prepare_temp_config_file(
                models_config, K
            )
            run_experiment(temp_config_file)
            os.remove(temp_config_file)
            self._unify_predictions_files(messy_preds_dir)
        except Exception as exc:
            raise exc
        finally:
            if clean_elliot_work_dir:
                shutil.rmtree(self.elliot_work_dir, ignore_errors=True)

            if clean_temp_dataset_files:
                self.clean_temp_dataset_files()

    @validate_arguments
    def clean_temp_dataset_files(self):
        """Cleans up temporary dataset files created for the experiments runs."""
        for temp_file in [
            self._temp_train_path,
            self._temp_test_path,
            self._temp_valid_path,
        ]:
            if temp_file is not None:
                os.remove(temp_file)
