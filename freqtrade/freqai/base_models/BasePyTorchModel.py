import logging
from abc import ABC, abstractmethod

import torch

from freqtrade.freqai.freqai_interface import IFreqaiModel
from freqtrade.freqai.torch.PyTorchDataConvertor import PyTorchDataConvertor


logger = logging.getLogger(__name__)


class BasePyTorchModel(IFreqaiModel, ABC):
    """
    Base class for PyTorch type models.
    User *must* inherit from this class and set fit() and predict() and
    data_convertor property.
    """

    def __init__(self, **kwargs):
        super().__init__(config=kwargs["config"])
        self.dd.model_type = "pytorch"
        training_settings = self.freqai_info.get("model_training_parameters", {})
        requested_device = str(training_settings.get("device", "auto")).strip().lower()
        if requested_device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("FreqAI PyTorch device must be auto, cpu, cuda, or mps")
        if requested_device == "cpu":
            self.device = "cpu"
        elif requested_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("FreqAI PyTorch device=cuda requested but CUDA is unavailable")
            self.device = "cuda"
        elif requested_device == "mps":
            if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
                raise RuntimeError("FreqAI PyTorch device=mps requested but MPS is unavailable")
            self.device = "mps"
        else:
            self.device = (
                "mps"
                if torch.backends.mps.is_available() and torch.backends.mps.is_built()
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )
        if self.device == "cpu":
            cpu_threads = int(training_settings.get("cpu_threads", torch.get_num_threads()))
            if cpu_threads < 1:
                raise ValueError("FreqAI PyTorch cpu_threads must be positive")
            torch.set_num_threads(cpu_threads)
        test_size = self.freqai_info.get("data_split_parameters", {}).get("test_size")
        self.splits = ["train", "test"] if test_size != 0 else ["train"]
        self.window_size = self.freqai_info.get("conv_width", 1)

    @property
    @abstractmethod
    def data_convertor(self) -> PyTorchDataConvertor:
        """
        a class responsible for converting `*_features` & `*_labels` pandas dataframes
        to pytorch tensors.
        """
        raise NotImplementedError("Abstract property")
