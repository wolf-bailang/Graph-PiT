import os

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union


@dataclass
class TrainConfig:
    root_dir = Path(__file__).parent.parent

    # Dataset path
    dataset_path: Union[Path, List[Path]] = os.path.join(root_dir, "datasets", "generated", "character_train")      # Path("./datasets1/generated/generated_things")
    # Validation dataset path
    val_dataset_path: Path = os.path.join(root_dir, "datasets", "generated", "character_val")      # Path("./datasets1/generated/generated_things_val")
    # The output directory where the model predictions and checkpoints will be written.
    output_dir: Path = os.path.join(root_dir, "training_results", "train_character")      # Path("./results/my_pit_model")
    # GPU device
    device: str = "cuda"
    # The resolution for input images, all the images will be resized to this size
    img_size: int = 1024
    # Batch size (per device) for the training dataloader
    train_batch_size: int = 2
    # Initial learning rate (after the potential warmup period) to use
    lr: float = 1e-5
    # Dataloader num workers.
    num_workers: int = 8
    # The beta1 parameter for the Adam optimizer.
    adam_beta1: float = 0.9
    # The beta2 parameter for the Adam optimizer
    adam_beta2: float = 0.999
    # Weight decay to use
    adam_weight_decay: float = 0.0  # 1e-2
    # Epsilon value for the Adam optimizer
    adam_epsilon: float = 1e-08
    # How often save images. Values less zero - disable saving
    log_image_frequency: int = 500
    # How often to run validation
    log_validation: int = 500
    # The number of images to save during each validation
    n_val_images: int = 50
    # A seed for reproducible training
    seed: Optional[int] = None
    # The number of accumulation steps to use
    gradient_accumulation_steps: int = 4
    # Whether to use mixed precision training
    mixed_precision: Optional[str] = "fp16"
    # Log to wandb
    report_to: Optional[str] = "wandb"
    # The number of training steps to run
    max_train_steps: int = 10000 # 1000000
    # Max grad for clipping
    max_grad_norm: float = 1.0
    # How often to save checkpoints
    checkpointing_steps: int = 100
    # The path to resume from
    resume_from_path: Optional[Path] = None # "./training_results/train_character/prior.ckpt"
    # The step to resume from, mainly for logging
    resume_from_step: Optional[int] = None
    # The path to the prior checkpoint
    prior_path: Optional[Path] = None
    # DiT number of layers
    num_layers: int = 8
    # DiT hidden dimensionality
    hidden_dim: int = 2048
    # DiT number of attention heads
    num_attention_heads: int = 32
    # Whether to use a reference grid
    use_ref: bool = False
    # Max number of crops
    max_crops: int = 3
    # Probability of converting to sketch
    sketch_prob: float = 0.0
    lambda_g: float =1.0
    lambda_r: float =1.0


