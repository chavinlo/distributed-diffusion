import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

import datasets
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from ....src.swarm.base import *

def main():
    config = get_config()
    tc = config['trainer_config']

    accelerator = Accelerator(
        mixed_precision=tc['global']['precision'],
    )

    set_seed(tc['global']['seed'])

    pretrained_model = tc['global']['pretrained_model_name_or_path']
    revision = tc['global']['revision']

    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model, subfolder="tokenizer", revision=revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model, subfolder="text_encoder", revision=revision
    )
    vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae", revision=revision)
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model, subfolder="unet", revision=revision
    )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if tc['global']['ema']:
        ema_unet = UNet2DConditionModel.from_pretrained(
            pretrained_model, subfolder="unet", revision=revision
        )
        ema_unet = EMAModel(ema_unet.parameters())

    #vram optimizations
    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()

    unet.enable_gradient_checkpointing()
    torch.backends.cuda.matmul.allow_tf32 = True

    try:
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.AdamW8bit
    except ImportError:
        raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

    learning_rate = (
            tc['global']['rate'] * tc['local']['final_batch_size']* accelerator.num_processes
            )
    
    optimizer = optimizer_cls(
        unet.parameters(),
        lr=learning_rate,
        betas=(tc['optimizer']['adam_beta1'], tc['optimizer']['adam_beta2']),
        weight_decay=tc['optimizer']['adam_weight_decay'],
        eps=tc['optimizer']['adam_epsilon'],
    )

    optimizer = wrap_opt(optimizer, config)