import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DatasetConfig:
    root: str
    split: str = "train2017"
    format: str = "yolo"


@dataclass
class ModelConfig:
    num_classes: int = 80
    hidden_dim: int = 256
    num_queries: int = 300
    use_uncertainty: bool = False
    backbone_name: str = "cspresnet"
    stem_channels: int = 32
    backbone_channels: list[int] = field(default_factory=lambda: [64, 128, 256])
    stage_blocks: list[int] = field(default_factory=lambda: [1, 2, 2])
    num_encoder_layers: int = 1
    num_decoder_layers: int = 3
    nhead: int = 8
    encoder_dim_feedforward: int | None = None
    decoder_dim_feedforward: int | None = None
    backbone_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class LossConfig:
    name: str = "default"
    task_aligner: str = "none"
    weights: dict[str, float] = field(default_factory=dict)


@dataclass
class TrainConfig:
    batch_size: int = 2
    lr: float = 1e-4
    epochs: int = 1


@dataclass
class Config:
    dataset: DatasetConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)


def load_config(path):
    path = Path(path)
    data = json.loads(path.read_text())
    dataset = DatasetConfig(**data["dataset"])
    model = ModelConfig(**data.get("model", {}))
    train = TrainConfig(**data.get("train", {}))
    loss = LossConfig(**data.get("loss", {}))
    return Config(dataset=dataset, model=model, train=train, loss=loss)
