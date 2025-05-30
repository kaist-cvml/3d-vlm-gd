import os
import sys
import math
import pickle
import types
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.utils as nn_utils
import torchvision.transforms as T
import torchvision.transforms.functional as VF

import hydra
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.absolute())
sys.path.append(project_root)

# Custom imports
from utils.functions import (
    fix_random_seeds,
    sigmoid,
    interpolate_features,
    query_pose_error,
    parse_yaml,
    preprocess_kps_pad,
    _fix_pos_enc,
)
from utils.tracking_metrics import compute_tapvid_metrics_for_video
from utils.tracking_model import ModelInference, Tracker

# ----------------
#     Pipeline
# ----------------
# ref stands for refactoring
from src.finetune_timm_mast3r import FinetuneMASt3RTIMM
from src.finetune_timm_vggt import FinetuneVGGTTIMM
from src.finetune_timm_me import FinetuneTIMM
from src.evaluate_timm import EvaluationCallback as TimmEvaluationCallback

from torch.utils.data import ConcatDataset
from data_utils.dataset_mast3r_objaverse import AugmentedCustomObjaverseDataset, ObjaverseMASt3RDataset
from data_utils.dataset_mast3r_scannetpp import AugmentedCustomScanNetPPDataset, ScanNetPPMASt3RDataset
from data_utils.dataset_vggt_objaverse import ObjaverseVGGTDataset
from data_utils.dataset_vggt_scannetpp import ScanNetPPVGGTDataset
from data_utils.dataset import ObjaverseCorrDataset, AugmentedDataset


def get_dataset(dataset_name, matcher_type):
    if dataset_name == 'scannetpp':
        if matcher_type == 'mast3r':
            return AugmentedCustomScanNetPPDataset(ConcatDataset([ScanNetPPMASt3RDataset()]), augmentation=False)
        elif matcher_type == 'vggt':
            return AugmentedCustomScanNetPPDataset(ConcatDataset([ScanNetPPVGGTDataset()]), augmentation=False)
    elif dataset_name == 'objaverse':
        if matcher_type == 'mast3r':
            return AugmentedCustomObjaverseDataset(ConcatDataset([ObjaverseMASt3RDataset('data/objaverse_renderings', 10_000)]))
        elif matcher_type == 'vggt':
            return AugmentedCustomObjaverseDataset(ConcatDataset([ObjaverseVGGTDataset('data/objaverse_renderings', 10_000)]))
        elif matcher_type == 'me':
            return AugmentedDataset(ConcatDataset([ObjaverseCorrDataset('data/objaverse_renderings', 10_000)]))

    return None

model = {
    "timm_mast3r": FinetuneMASt3RTIMM,
    "timm_vggt": FinetuneVGGTTIMM,
    "timm_me": FinetuneTIMM,
}


@hydra.main(
    config_path='../config',
    config_name='finetune_timm_mast3r_scannetpp',
    # config_name='finetune_timm_vggt_scannetpp',
    # config_name='finetune_timm_mast3r_objaverse',
    # config_name='finetune_timm_vggt_objaverse',
    # config_name='finetune_timm_me_objaverse',
    version_base='1.2',
)
def main(cfg):
    # Add evaluation methods as a list in the config
    eval_methods = cfg.get(
        'evaluation_methods', ['semantic_transfer']
    )

    # is_dev_mode = True
    is_dev_mode = False
    limit_batches = 2 if is_dev_mode else None
    # limit_batches = 1 if is_dev_mode else None  # Naive CLIP
    # ---------------------------------------

    fix_random_seeds()
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']

    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger = TensorBoardLogger(
        output_dir,
        name='experiment_logs',
        version=f"{start_time}_{cfg.get('experiment_version', 'v1')}",
        default_hp_metric=False
    )
    logger.log_hyperparams({
        'notes': 'Your experiment notes here',
        'start_time': start_time,
        'model_type': cfg.backbone,
        'dataset': cfg.dataset,
        'batch_size': cfg.batch_size if hasattr(cfg, 'batch_size') else 'default'
    })


    model_class = model[f"{cfg.model}_{cfg.matcher}"]
    dataset_instance = get_dataset(cfg.dataset, cfg.matcher)
    pl_module = model_class(r=4, backbone_size=cfg.backbone, datasets=dataset_instance)

    if cfg.model == 'timm':
        evaluation_callback = TimmEvaluationCallback(
            cfg,
            eval_every_n_epochs=10,
            eval_methods=eval_methods
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'checkpoints', start_time),
        filename='{epoch:02d}-{val_loss:.2f}',
        save_last=True,
        every_n_epochs=1,
        save_top_k=-1,
        verbose=True
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=-1,
        strategy='ddp_find_unused_parameters_true',
        max_epochs=500,
        limit_train_batches=limit_batches,
        gradient_clip_val=1.0,
        logger=logger,
        callbacks=[
            evaluation_callback,
            checkpoint_callback,
        ],
    )

    trainer.fit(pl_module)
    pass


if __name__ == '__main__':
    main()