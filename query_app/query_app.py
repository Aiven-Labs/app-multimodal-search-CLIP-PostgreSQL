#!/usr/bin/env python3

"""An app to find (the first four) images matching a text string, and display them.
"""

import logging
import os

from typing import Annotated, List, Literal

import psycopg

import httpx

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

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

# Get our model name
MODEL_NAME = os.environ.get('MODEL_NAME', 'openai/clip-vit-base-patch32')

# Get the URL for our CLIP embedding service
CLIP_SERVICE_URL = os.environ.get('CLIP_SERVICE_URL', 'http://localhost:8000')


# And the response
class ItemResult(BaseModel):
    embedding: List[float]
    length: int

class EmbeddingResponse(BaseModel):
    embeddings: List[ItemResult]
    count: int


async def get_text_embedding(text):
    try:
        response = httpx.post(
            f'{CLIP_SERVICE_URL}/embed',
            json={
                "model_name": MODEL_NAME,
                "datatype": "text",
                "values": [text],
            },
        )
        logger.info(f"Response from CLIP: {response.text}")
        response.raise_for_status()

        data = response.json()
        return data["embeddings"][0]["embedding"]
    except Exception as exc:
        logger.error(f'Error getting text embedding from {CLIP_SERVICE_URL}: {exc.__class__.__name__}: {exc}')
        #raise Exception('Unable to get text embedding')
        raise Exception(f'Error getting text embedding from {CLIP_SERVICE_URL}: {exc.__class__.__name__}: {exc}')


def vector_to_string(embedding):
    """Convert our (ndarry) embedding vector into a string that SQL can use.
    """
    vector_str = ", ".join(str(x) for x in embedding)
    vector_str = f'[{vector_str}]'
    return vector_str


async def search_for_matches(text):
    """Returns pairs of the form (image_name, image_url)"""
    logger.info(f'Searching for {text!r}')
    vector = await get_text_embedding(text)

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
        logger.error(f'{exc.__class__.__name__}: {exc}')
        raise Exception(f'Unable to query database')
        # I tried including the actual exception in the raised error, which is
        # what the user will see on the query page, but it's at best confusing,
        # so let's not do that
        #raise Exception(f'Error querying database: {exc}')


app = FastAPI(redirect_slashes=False)
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

    try:
        results = await search_for_matches(search_text)
    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="images.html",
            context={
                "images": [],
                "error_message": str(e),
            }
        )

    return templates.TemplateResponse(
        request=request,
        name="images.html",
        context={
            "images": results,
            "error_message": "",
        }
    )
