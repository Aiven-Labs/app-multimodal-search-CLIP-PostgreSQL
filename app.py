#!/usr/bin/env python3

"""An app to find (the first four) images matching a text string, and display them.
"""

import asyncio
import logging
import os

from typing import Annotated, Callable, Union

import clip
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


LOCAL_MODEL = Path('./models/ViT-B-32.pt').absolute()
MODEL_NAME = 'ViT-B/32'
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
    # If we download it remotely, it will default to being cached in ~/.cache/clip
    try:
        if LOCAL_MODEL.exists():
            logger.info(f'Importing CLIP model {MODEL_NAME} from {LOCAL_MODEL.parent}')
            logger.info(f'Using {DEVICE}')
            clip_model.model, clip_model.preprocess = clip.load(MODEL_NAME, device=DEVICE, download_root=LOCAL_MODEL.parent)
        else:
            logger.info(f'Importing CLIP model {MODEL_NAME}')
            logger.info(f'Using {DEVICE}')
            clip_model.model, clip_model.preprocess = clip.load(MODEL_NAME, device=DEVICE)
    except Exception as exc:
        clip_model.error_string = 'Unable to load CLIP model - please restart the application'
        logger.exception(clip_model.error_string)
    else:
        logger.info('CLIP model imported')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Define events at the start and end of the app lifespan

    This means we'll start the CLIP model loaded before the app appears for the user,
    but won't block them from seeing the web page early.
    """
    logger.info('Async load task starting')
    blocking_loader = asyncio.to_thread(load_clip_model)
    asyncio.create_task(blocking_loader)
    ###load_clip_model()
    yield
    # We don't have an unload step


def get_single_embedding(text):
    with torch.no_grad():
        # Encode the text to compute the feature vector and normalize it
        text_input = clip.tokenize([text]).to(DEVICE)
        text_features = clip_model.model.encode_text(text_input)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # Return the feature vector
    return text_features.cpu().numpy()[0]


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
