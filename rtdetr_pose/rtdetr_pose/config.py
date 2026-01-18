import json
from dataclasses import dataclass, field
from pathlib import Path


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


def load_config(path):
    path = Path(path)
    data = json.loads(path.read_text())
    dataset = DatasetConfig(**data["dataset"])
    model = ModelConfig(**data.get("model", {}))
    train = TrainConfig(**data.get("train", {}))
    return Config(dataset=dataset, model=model, train=train)
