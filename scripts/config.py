# config.py
import torch

IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 10
SAMPLES_PER_EPOCH = 3000
LR = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Loss weights
LAMBDA_BINARY = 1.0
LAMBDA_RANK = 0.5

# Recall-biased threshold
ACTIVE_THRESHOLD = 0.4

# Fine-tuning
FREEZE_BACKBONES_EPOCHS = 5   # warmup
UNFREEZE_LR = 1e-5

# Paths
RUN_NAME = "effresvit_listnet_v1"
CKPT_DIR = f"checkpoints/{RUN_NAME}"
