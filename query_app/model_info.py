#!/usr/bin/env python3

"""Gather the information about our model and where to store it

Usage is intended to be `from model_support import *`

Notes on model openai/clip-vit-base-patch32:

* See https://huggingface.co/openai/clip-vit-base-patch32 for the "model card"
  which describes it

* See https://huggingface.co/openai/clip-vit-base-patch32/tree/main for a list
  of the files in the model

* Note that this model does not support safetensors
  (and specifically it does not include a `model.safetensors` file).

  HuggingFace recommend using safetensors instead of the pickled `.bin` file
  that has traditionally been used with PyTorch, because of the security problems
  of pickle files. See
  https://huggingface.co/docs/diffusers/v0.25.0/en/using-diffusers/using_safetensors#load-safetensors

  For a list of models that *do* support safetensors, see
  https://huggingface.co/models?library=safetensors
"""

import os

from pathlib import Path

MODEL_NAME = os.environ.get('MODEL_NAME', 'openai/clip-vit-base-patch32')

MODEL_DIRNAME = os.environ.get('MODEL_DIR', f'./models/{MODEL_NAME}')
CACHE_DIRNAME = os.environ.get('CACHE_DIR', './models/cache')

MODEL_DIR = Path(MODEL_DIRNAME).absolute()
CACHE_DIR = Path(CACHE_DIRNAME).absolute()
