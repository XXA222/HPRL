"""Built-in FreqAI PyTorch multitask model for Hedge directional and risk targets."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from freqtrade.freqai.base_models.BasePyTorchRegressor import BasePyTorchRegressor
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.hedge_rl.networks import HedgeMultiTaskMLP
from freqtrade.freqai.torch.PyTorchDataConvertor import (
    DefaultPyTorchDataConvertor,
    PyTorchDataConvertor,
)
from freqtrade.freqai.torch.PyTorchModelTrainer import PyTorchModelTrainer


class HedgePyTorchMultiTaskRegressor(BasePyTorchRegressor):
    """Train one neural network for five coordinated Hedge outputs."""

    @property
    def data_convertor(self) -> PyTorchDataConvertor:
        return DefaultPyTorchDataConvertor(target_tensor_type=torch.float)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        settings = self.freqai_info.get("model_training_parameters", {})
        self.learning_rate = float(settings.get("learning_rate", 3e-4))
        self.model_kwargs: dict[str, Any] = settings.get("model_kwargs", {})
        self.trainer_kwargs: dict[str, Any] = settings.get("trainer_kwargs", {})

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        n_features = data_dictionary["train_features"].shape[-1]
        n_labels = data_dictionary["train_labels"].shape[-1]
        if n_labels != 5:
            raise ValueError(
                "HedgePyTorchMultiTaskRegressor requires exactly five Hedge target columns"
            )
        model = HedgeMultiTaskMLP(
            input_dim=n_features,
            output_dim=n_labels,
            **self.model_kwargs,
        ).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        trainer = self.get_init_model(dk.pair)
        if trainer is None:
            trainer = PyTorchModelTrainer(
                model=model,
                optimizer=optimizer,
                # FreqAI normalizes every target before training.  Use a regression
                # loss in normalized label space; bounded Hedge transforms belong to
                # standalone training/inference, not before label inverse-transform.
                criterion=torch.nn.SmoothL1Loss(),
                device=self.device,
                data_convertor=self.data_convertor,
                tb_logger=self.tb_logger,
                **self.trainer_kwargs,
            )
        trainer.fit(data_dictionary, self.splits)
        return trainer

    def predict(self, unfiltered_df, dk: FreqaiDataKitchen, **kwargs):
        dk.find_features(unfiltered_df)
        filtered_df, _ = dk.filter_features(
            unfiltered_df, dk.training_features_list, training_filter=False
        )
        prediction_features, outliers, _ = dk.feature_pipeline.transform(
            filtered_df, outlier_check=True
        )
        x = self.data_convertor.convert_x(prediction_features, device=self.device)
        self.model.model.eval()
        with torch.no_grad():
            # Predictions remain in the normalized label space expected by the
            # FreqAI label pipeline.  Applying sigmoid/tanh/softplus here would
            # corrupt inverse scaling and bias all five targets.
            output = self.model.model(x)
        pred_df = pd.DataFrame(output.cpu().numpy(), columns=dk.label_list)
        pred_df, _, _ = dk.label_pipeline.inverse_transform(pred_df)
        if self.ft_params.get("DI_threshold", 0) > 0:
            dk.DI_values = dk.feature_pipeline["di"].di_values
        else:
            dk.DI_values = np.zeros(outliers.shape[0])
        dk.do_predict = outliers
        return pred_df, dk.do_predict
