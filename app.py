#!/usr/bin/env python3

"""An app to find (the first four) images matching a text string, and display them.
"""

import asyncio
import logging
import os

from typing import Annotated, Callable, Union

import psycopg
import torch
import PIL

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import CLIPProcessor, CLIPModel

from model_info import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
)

logger = logging.getLogger(__name__)

SERVICE_URI = os.getenv("DATABASE_URL")
if not SERVICE_URI:
    # Try the .env file
    load_dotenv()
    SERVICE_URI = os.getenv("DATABASE_URL")
# At which point we rather hope we found the URL for our PG database...


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Model:
    """The types of the values are found from the docstring for clip.load

    See also the source code at https://github.com/openai/CLIP/blob/main/clip/clip.py

    (we could just make them type Any, but it's interesting to know the actual types)
    """
    model: Union[None, torch.nn.Module]
    preprocess: Union[None, Callable[[PIL.Image], torch.Tensor]]
    error_string: str

clip_model = Model(None, None, "CLIP model not loaded yet - try again soon")


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
        clip_model.error_string = 'Unable to load CLIP model - please restart the application'
        logger.exception(clip_model.error_string)
    else:
        logger.info('CLIP model imported')
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


def get_single_embedding(text):
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

    # Return the feature vector (there is just the one)
    return features.numpy()[0]


def vector_to_string(embedding):
    """Convert our (ndarry) embedding vector into a string that SQL can use.
    """
    vector_str = ", ".join(str(x) for x in embedding.tolist())
    vector_str = f'[{vector_str}]'
    return vector_str


def search_for_matches(text):
    """Returns pairs of the form (image_name, image_url)"""
    logger.info(f'Searching for {text!r}')
    vector = get_single_embedding(text)

    embedding_string = vector_to_string(vector)

    # Perform search
    try:
        with psycopg.connect(SERVICE_URI) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT filename, url FROM pictures ORDER BY embedding <-> %s LIMIT 4;",
                    (embedding_string,),
                )
                return cur.fetchall()
    except Exception as exc:
        print(f'{exc.__class__.__name__}: {exc}')
        return []


app = FastAPI(lifespan=lifespan, redirect_slashes=False)
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "search_hint": "Find images like...",
        },
    )


@app.post("/search_form", response_class=HTMLResponse)
async def search_form(request: Request, search_text: Annotated[str, Form()]):
    logging.info(f'Search form requests {search_text!r}')
    if not clip_model.model:
        return templates.TemplateResponse(
            request=request,
            name="images.html",
            context={
                "images": [],
                "error_message": clip_model.error_string,
            }
        )

    results = search_for_matches(search_text)
    return templates.TemplateResponse(
        request=request,
        name="images.html",
        context={
            "images": results,
            "error_message": "",
        }
    )
