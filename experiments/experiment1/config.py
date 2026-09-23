from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Config:
    data_dir: str = "Dataset"
    output_dir: str = "output"
    seed: int = 42
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    img_size: int = 224
    filter_mode: str = "bilateral"
    augment: bool = True
    rotation: int = 20
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.1
    fe_arch: str = "effnet_cbam"
    fe_epochs: int = 40
    fe_lr: float = 0.001
    fe_weight_decay: float = 0.0005
    fe_momentum: float = 0.9
    fe_dropout: float = 0.5
    fe_dropout2: float = 0.25
    fe_batch: int = 32
    fe_num_workers: int = 2
    n_features: int = 1280
    n_selected: int = 500
    fs_method: str = "shap"
    num_round: int = 100
    cls_lr: float = 0.001
    cls_epochs: int = 200
    cls_batch: int = 32
    cls_hidden: int = 256
    cls_dropout: float = 0.5
    cls_dropout2: float = 0.3

    def save(self, path: str | Path):
        import json

        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path):
        import json

        return cls(**json.loads(Path(path).read_text()))