import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from pydantic import Field, FilePath, NonNegativeInt, PositiveInt, PrivateAttr
from recbole.config import Config
from recbole.data.interaction import Interaction
from recbole.data.utils import create_dataset, data_preparation
from recbole.model.abstract_recommender import AbstractRecommender
from recbole.trainer import Trainer

from ..base_rec import BaseRec


class RecBoleRec(BaseRec):
    """
    RecBoleRec is a recommendation system class built on top of the RecBole framework.
    It manages dataset preparation, model training, and recommendation generation.
    """

    valid_path: FilePath = Field(
        ...,
        description="Path to the validation data.",
    )
    rating_thr: NonNegativeInt = Field(
        int=0,
        description="Rating threshold for determining positive labels.",
    )
    _dataset_name = PrivateAttr(default="temp_recbole")
    _temp_dir_path = PrivateAttr(default=None)
    _model_config = PrivateAttr()

    class Config:
        extra = "forbid"

    def model_post_init(self, __context):
        super().model_post_init(__context)

        self.prepare_temp_dataset()

    @staticmethod
    def _expand_batch_for_items(initial_batch: Interaction, item_ids: list):
        """
        This method takes an initial RecBole batch and constructs a dictionary
        containing the expanded batch, where each entry is repeated for every
        item ID provided in the `item_ids` list. The dictionary also includes
        a `test_item_id` key, which stores the repeated `item_id` values from
        the original batch.
        """
        batch_size = initial_batch["user_id"].shape[0]
        num_items = len(item_ids)

        new_batch = {
            key: torch.repeat_interleave(torch.tensor(value), num_items, dim=0)
            for key, value in initial_batch.numpy().items()
        }

        repeated_item_ids = torch.tensor(
            item_ids, dtype=initial_batch["item_id"].dtype
        ).repeat(batch_size)
        new_batch["item_id"] = repeated_item_ids

        test_item_ids = torch.repeat_interleave(
            initial_batch["item_id"], num_items, dim=0
        )
        new_batch["test_item_id"] = test_item_ids

        return new_batch

    @staticmethod
    def _split_batch(batch: dict, batch_size: PositiveInt):
        """
        This method splits a given batch into smaller batches of the specified `batch_size`.
        Each smaller batch is a dictionary containing a subset of the original batch's data.
        """
        total_samples = batch["user_id"].size(0)
        num_batches = (total_samples + batch_size - 1) // batch_size

        return [
            {
                key: value[i * batch_size : min((i + 1) * batch_size, total_samples)]
                for key, value in batch.items()
            }
            for i in range(num_batches)
        ]

    def prepare_temp_dataset(self):
        """
        Prepares a temporary dataset by reformatting and storing it in a structured format
        compatible with the RecBole framework. The method renames columns, reorganizes data,
        and saves the processed dataset in the appropriate format for training and evaluation.
        """

        self._logger.info("Reformatting the dataset...")
        self._temp_dir_path = tempfile.mkdtemp()
        temp_data_path = f"{self._temp_dir_path}/{self._dataset_name}"
        Path(temp_data_path).mkdir(exist_ok=True, parents=True)

        dataset_mapping = {
            "train": self.train_path,
            "test": self.test_path,
            "valid": self.valid_path,
        }

        for dataset_type, dataset_file_path in dataset_mapping.items():
            df = pd.read_csv(
                dataset_file_path,
                sep=self.dataset_sep,
                compression=self.dataset_compression,
            )

            column_rename_dict = {
                df.columns[self.dataset_user_idx]: "user_id:token",
                df.columns[self.dataset_item_idx]: "item_id:token",
                df.columns[self.dataset_rating_idx]: "rating:float",
            }
            context_columns = [
                col for col in df.columns if col not in column_rename_dict
            ]
            context_rename_dict = {
                col: f"c{i + 1}:token" for i, col in enumerate(context_columns)
            }
            column_rename_dict.update(context_rename_dict)
            df.rename(columns=column_rename_dict, inplace=True)

            output_name = f"{temp_data_path}/{self._dataset_name}.{dataset_type}.inter"
            df.to_csv(output_name, sep="\t", index=False)

    def recommend(
        self,
        recbole_model: AbstractRecommender,
        output_path: str,
        K: Optional[PositiveInt] = None,
        extra_config: Optional[dict] = None,
        clean_temp_dir: bool = False,
    ):
        """
        Generates recommendations using the specified RecBole model.

        This method loads the dataset, trains the model, prepares test batches, and
        computes predictions. The top-K recommendations are extracted and saved to the
        specified output path.

        Args:
            `recbole_model`: The RecBole model to be used for recommendations.
            `output_path`: Path to save the recommendation results.
            `K`: Number of top-ranked recommendations to retrieve.
            `extra_config`: Additional RecBole configuration parameters for the model.
            `clean_temp_dir`: Whether to remove the temporary dataset directory after execution.
        """
        try:
            if self._temp_dir_path is None:
                self.prepare_temp_dataset()

            self._logger.info("Preparing the model configuration...")
            benchmark_filename = ["train", "valid", "test"]

            parameter_dict = {
                "data_path": self._temp_dir_path,
                "benchmark_filename": benchmark_filename,
                "load_col": None,
                "threshold": {"rating": self.rating_thr},
                **(extra_config if extra_config is not None else {}),
            }
            model_config = Config(
                model=recbole_model,
                dataset=self._dataset_name,
                config_dict=parameter_dict,
            )

            self._logger.info("Loading the dataset...")
            dataset = create_dataset(model_config)
            train_data, valid_data, test_data = data_preparation(model_config, dataset)

            self._logger.info("Loading the model...")
            loaded_model = recbole_model(model_config, train_data.dataset).to(
                model_config["device"]
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)

                self._logger.info("Training the model...")
                trainer = Trainer(model_config, loaded_model)
                trainer.fit(train_data, valid_data)

                self._logger.info("Preparing the batches...")
                dataset_items = dataset.get_item_feature().numpy()["item_id"]
                dataset_nitems = dataset_items.shape[0]

                if K is None:
                    K = dataset_nitems

                K = min(K, dataset_nitems)

                full_batch = self._expand_batch_for_items(
                    test_data.dataset.inter_feat,
                    dataset_items,
                )

                batches = self._split_batch(full_batch, dataset_nitems)

                self._logger.info("Prediction starts.")

                num_predictions = test_data.dataset.inter_feat["user_id"].shape[0] * K
                np_user_ids = np.empty(num_predictions, dtype=object)
                np_item_ids = np.empty(num_predictions, dtype=object)
                np_scores = np.empty(num_predictions, dtype=np.float32)
                np_test_item_ids = np.empty(num_predictions, dtype=object)
                position = 0

                for i, batch in enumerate(batches):
                    if not i % 1000:
                        self._logger.info("Batch {} / {}".format(i, len(batches)))

                    # Make sure the batch is processed at the same device as the model
                    batch_device = {
                        key: value.to(loaded_model.device)
                        for key, value in batch.items()
                    }

                    preds = loaded_model.predict(batch_device)
                    scores, indices = torch.topk(preds, K)
                    np_indices = indices.cpu().numpy()

                    np_user_ids[position : position + K] = dataset.id2token(
                        "user_id", batch_device["user_id"].cpu().numpy()[np_indices]
                    )
                    np_item_ids[position : position + K] = dataset.id2token(
                        "item_id", batch_device["item_id"].cpu().numpy()[np_indices]
                    )
                    np_scores[position : position + K] = scores.cpu().detach().numpy()
                    np_test_item_ids[position : position + K] = dataset.id2token(
                        "item_id",
                        batch_device["test_item_id"].cpu().numpy()[np_indices],
                    )
                    position += K

                    # Free memory from the device
                    del batch_device

            self._logger.info("Saving predictions...")
            Path(output_path).parent.mkdir(exist_ok=True, parents=True)
            preds_df = pd.DataFrame(
                {
                    self.preds_user_col_name: np_user_ids,
                    self.preds_item_col_name: np_item_ids,
                    self.preds_score_col_name: np_scores,
                    self.preds_test_item_col_name: np_test_item_ids,
                }
            )
            preds_df = preds_df.loc[preds_df[self.preds_item_col_name] != "[PAD]"]

            preds_df.to_csv(
                output_path,
                sep=self.preds_sep,
                compression=self.preds_compression,
                index=False,
            )
        except Exception as exc:
            raise exc
        finally:
            if clean_temp_dir:
                self.clean_temp_dir()

    def clean_temp_dir(self):
        """Removes the temporary dataset directory if it exists."""

        if self._temp_dir_path is None:
            return

        self._logger.info("Removing temporal dataset...")
        shutil.rmtree(self._temp_dir_path, ignore_errors=True)
        self._temp_dir_path = None
