"""HyperGen: Unified post-training and inference framework for diffusion models."""

from typing import TYPE_CHECKING

from hypergen.core.model import Model
from hypergen.core.dataset import Dataset
from hypergen.server.api import serve

if TYPE_CHECKING:
    from hypergen.core.lora import LoRAAdapter

__version__ = "0.1.0"
__all__ = ["Model", "Dataset", "serve"]