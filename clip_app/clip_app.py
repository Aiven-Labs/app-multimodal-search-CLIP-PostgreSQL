#!/usr/bin/env python3

"""An app ask a local CLIP model for an embedding and return it.
"""

import asyncio
import logging
import json

from contextlib import asynccontextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable, List, Literal, Sequence, Union
from urllib.parse import urlparse

import httpx
import torch

from PIL import Image
from PIL.ImageFile import ImageFile
from fastapi import status
from fastapi import FastAPI, Request
from fastapi import HTTPException
from pydantic import BaseModel
from transformers import CLIPProcessor, CLIPModel

# Get our model name and directories
from model_info import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
)

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Model:
    """The types of the values are found from the docstring for clip.load

    See also the source code at https://github.com/openai/CLIP/blob/main/clip/clip.py

    (we could just make them type Any, but it's interesting to know the actual types)
    """
    model: Union[None, torch.nn.Module]
    preprocess: Union[None, Callable[[Image], torch.Tensor]]
    error_string: str

clip_model = Model(None, None, f"CLIP model {MODEL_NAME} not loaded yet - try again soon")


def load_clip_model():
    """Load the open CLIP model"""

    logger.info(f'Using device {DEVICE} for model calculations')

    try:
        # Load the open CLIP model
        # If we're being run from our Dockerfile, then the model should already
        # have been downloaded to MODEL_DIR, so let's check for that first.
        # If that directory doesn't exist, fall back to the normal "download and
        # cache" approach, when the model will be cached in ~/.cache/clip.
        if MODEL_DIR.exists():
            logger.info(f'Importing CLIP model {MODEL_NAME} from {MODEL_DIR}')
            clip_model.model = CLIPModel.from_pretrained(MODEL_DIR).to(DEVICE)
            clip_model.processor = CLIPProcessor.from_pretrained(MODEL_DIR)
        else:
            logger.info(f'Importing CLIP model {MODEL_NAME} from HuggingFace')
            clip_model.model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
            clip_model.processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    except Exception as exc:
        clip_model.error_string = f'Unable to load CLIP model {MODEL_NAME}: Please restart the application'
        logger.exception(clip_model.error_string)
    else:
        logger.info(f'CLIP model {MODEL_NAME} imported')
        # We're not expecting this to show up if the model is loaded,
        # but the original message would be misleading
        clip_model.error_string = 'Something unexpected has gone wrong'


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Define events at the start and end of the app lifespan

    This means we'll start the CLIP model loaded before the app appears for the user,
    but won't block them from seeing the web page early.
    """
    logger.info('Async load task starting')
    blocking_loader = asyncio.to_thread(load_clip_model)
    asyncio.create_task(blocking_loader)
    yield
    # We don't have an unload step


async def get_image_data(url: str) -> ImageFile:
    """Load image data from a URL.

    We assume a "file:" URL for a local file, or an "http:" or "https:" URL
    for remote data.
    """
    if url.startswith('file:'):
        parsed_url = urlparse(url)
        file_path = Path(parsed_url.path)
        return Image.open(file_path)

    async with httpx.AsyncClient() as client:
        # Retrieve the URL
        response = await client.get(
            url,
            follow_redirects=True,  # For instance, we know that we use GitHub URLs that redirect
        )
        response.raise_for_status()

        # Turn the bytes into a "file like" object for PIL
        image_bytes = BytesIO(response.content)

        return Image.open(image_bytes)


def get_image_embedding(image_data: ImageFile) -> List[float]:
    with torch.no_grad():
        inputs = clip_model.processor(
            images=[image_data],
            return_tensors='pt',
            padding=True,           # do we need this?
        ).to(DEVICE)

        # Compute the feature vectors
        features = clip_model.model.get_image_features(**inputs)

        # Normalise the embeddings, to make them easier to compare
        features /= features.norm(dim=-1, keepdim=True)

    # Return the feature vector
    return features.numpy()[0].tolist()


def get_text_embedding(text: str) -> List[float]:
    with torch.no_grad():
        inputs = clip_model.processor(
            text=[text],
            return_tensors='pt',
            padding=True,           # do we need this?
        ).to(DEVICE)

        # Compute the feature vectors
        features = clip_model.model.get_text_features(**inputs)

        # Normalise the embeddings, to make them easier to compare
        features /= features.norm(dim=-1, keepdim=True)

    # Return the feature vector
    return features.numpy()[0].tolist()


# Our query structure
class EmbeddingRequest(BaseModel):
    model_name: str
    datatype: Literal["text", "image"]
    value: str


class EmbeddingResponse(BaseModel):
    embedding: List[float]

app = FastAPI(lifespan=lifespan, redirect_slashes=False)


# Add a health check endpoint for use by compose
@app.get("/health")
async def healthcheck():
    logger.info('Healthcheck requested')
    return {"status": "ok"}


# Example usage:
#   curl -X 'POST' \
#   'http://127.0.0.1:8000/embed' \
#   -i \
#   -H 'Content-Type: application/json' \
#   -d '{
#   "model_name": "openai/clip-vit-base-patch32",
#   "datatype": "text",
#   "value": "Man jumping"
#   }'
# (I like including `-i` (`--include`) to show the status code, although
# it does show other stuff I don't normally want as much. Approaches to
# show *just* the status code and response are generally clunkier.)
#
@app.post("/embed", response_model=EmbeddingResponse)
async def process_embedding_request(payload: EmbeddingRequest) -> EmbeddingResponse:
    """Given a list of text prompts or of image file URLs, return the corresponding embeddings.
    """
    logger.info(f'Processing embedding request: {payload}')
    if payload.model_name != MODEL_NAME:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,  # essentially a validation error
            detail=f'Current model is {MODEL_NAME}, {payload.model_name} is not available',
        )
    if not clip_model.model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=clip_model.error_string,
        )

    embeddings = []
    if payload.datatype == "text":
        embedding = get_text_embedding(payload.value)
    elif payload.datatype == "image":
        image_data = await get_image_data(payload.value)
        embedding = get_image_embedding(image_data)
    else:
        # This should never occur, but just in case...
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'"datatype" {payload.datatype} is neither text nor image',
        )

    result = {"embedding": embedding}

    logger.info(f'Rough response size estimate {len(json.dumps(result))}')

    return result


