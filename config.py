from dataclasses import dataclass

@dataclass
class Config:
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 4
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 5

    # Paper settings
    wavelet: str = "db4"
    wavelet_levels: int = 3
    pca_variance: float = 0.95
    effnet_dim: int = 128
    vit_dim: int = 256
    fusion_dim: int = 256
    fusion_heads: int = 4

    art_rho: float = 0.85
    art_beta: float = 0.5
    final_threshold: float = 0.47

    seed: int = 12345
