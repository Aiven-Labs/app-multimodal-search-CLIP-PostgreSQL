#!/usr/bin/env python3

"""Download the CLIP model we use, so that the app does not need to do so.
"""

import logging
import shutil

##from transformers import AutoModel, AutoTokenizer
from huggingface_hub import snapshot_download

# Get our model name and directories
from model_info import *


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
)

logger = logging.getLogger(__name__)

logging.info(f'Getting {MODEL_NAME} to {MODEL_DIR}')
# For the openai/clip-vit-base-patch32 model
# (see https://huggingface.co/openai/clip-vit-base-patch32/tree/main)
# this retrieves 589M, mostly consisting of the `pytorch_model.bin`
# file which is 578M.
# It also downloads about 4K of `ref` files to
#   ~/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/
# which is small enough we can probably ignore them
#
# Remember that the exclusions may need changing for a different model,
#
# If the model is used with `use_safetensors=True` (which is not supported
# for openai/clip-vit-base-patch32), then we'd need to keep the `*.h5` files
# (and maybe `*.msgpack`?), but instead ignore `*.bin`
#
# For a list of models that *do* support safe tensors, see
# https://huggingface.co/models?library=safetensors
#
# For some advice on using `snapshot_download` see
#   https://huggingface.co/docs/huggingface_hub/en/guides/download#filter-files-to-download
snapshot_download(
    MODEL_NAME,
    local_dir=MODEL_DIR,
    ignore_patterns=['*.h5', '*.msgpack', '*.txt', '.gitattributes'],
)

logging.info('Done')
