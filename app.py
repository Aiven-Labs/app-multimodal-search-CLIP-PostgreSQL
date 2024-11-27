#!/usr/bin/env python3

"""An app to find (the first four) images matching a text string, and display them.
"""

import logging
import os

from typing import Annotated, Union

import clip
import psycopg
import torch

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Our images are kept locally, so make them available to the `img` tag
app.mount("/photos", StaticFiles(directory="photos"), name="photos")


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
)

logger = logging.getLogger(__name__)

load_dotenv()
SERVICE_URI = os.getenv("PG_SERVICE_URI")

# Load the open CLIP model
logger.info('Importing CLIP model')
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f'Using {device}')
model, preprocess = clip.load("ViT-B/32", device=device)


def get_single_embedding(text):
    with torch.no_grad():
        # Encode the text to compute the feature vector and normalize it
        text_input = clip.tokenize([text]).to(device)
        text_features = model.encode_text(text_input)
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
    logger.info(f'Searching for {text!r}')
    vector = get_single_embedding(text)

    embedding_string = vector_to_string(vector)

    # Perform search
    try:
        with psycopg.connect(SERVICE_URI) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM pictures ORDER BY embedding <-> %s LIMIT 4;",
                    (embedding_string,),
                )
                rows = cur.fetchall()
                return [row[0] for row in rows]
    except Exception as exc:
        print(f'{exc.__class__.__name__}: {exc}')
        return []





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
    images = search_for_matches(search_text)
    return templates.TemplateResponse(
        request=request,
        name="images.html",
        context={
            "images": images,
        }
    )
