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
from fastapi import status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
)

logger = logging.getLogger(__name__)

# httpx will log all GET and POST requests at level INFO, which is a bit much,
# so let's disable that
logging.getLogger("httpx").setLevel(logging.ERROR)

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    # Try the .env file
    load_dotenv()
    DATABASE_URL = os.getenv("DATABASE_URL")
# At which point we rather hope we found the URL for our PG database...

# Get our model name
MODEL_NAME = os.environ.get('MODEL_NAME', 'openai/clip-vit-base-patch32')

# Get the URL for our CLIP embedding service
CLIP_SERVICE_URL = os.environ.get('CLIP_SERVICE_URL', 'http://localhost:8000')

# Our table name
TABLE_NAME = 'pictures'


async def clip_service_is_ready() -> False:
    """Is the CLIP service ready to serve embeddings?
    """
    logger.info('Checking CLIP service')
    try:
        response = httpx.get(
            f'{CLIP_SERVICE_URL}/healthy',
            timeout=None,    # the default is documented as 5 seconds
        )
        if response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
            return False

        logger.info('CLIP service is ready')
        return True

    except Exception as exc:
        logger.error(f'Error getting CLIP service readiness from {CLIP_SERVICE_URL}: {exc.__class__.__name__}: {exc}')
        # raise Exception(f'Error getting CLIP service readiness from {CLIP_SERVICE_URL}: {exc.__class__.__name__}: {exc}')
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Define events at the start and end of the app lifespan

    This means we can start the query app before the CLIP service ready,
    and still give _some_ response to the user
    """
    logger.info('Async load task starting')
    blocking_loader = asyncio.to_thread(wait_for_clip_model)
    asyncio.create_task(blocking_loader)
    yield
    # We don't have an unload step


async def get_text_embedding(text) -> List[float]:
    logger.info(f'Requesting embeddings for {text}')
    try:
        response = httpx.post(
            f'{CLIP_SERVICE_URL}/embed',
            json={
                "model_name": MODEL_NAME,
                "datatype": "text",
                "value": text,
            },
        )
        response.raise_for_status()

        data = response.json()
        return data["embedding"]
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
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT filename, url FROM {TABLE_NAME} ORDER BY embedding <-> %s LIMIT 4;",
                    (embedding_string,),
                )
                return cur.fetchall()
    except Exception as exc:
        logger.error(f'Unable to query database {exc.__class__.__name__}: {exc}')
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

    if not await clip_service_is_ready():
        return templates.TemplateResponse(
            request=request,
            name="images.html",
            context={
                "images": [],
                "error_message": "CLIP service is not ready to serve embeddings yet - please try again later",
            }
        )

    # It would also be nice to be able to check if the database is populated yet.
    # We _could_ COUNT the records in the database, and report how many / complain if
    # there aren't enough, or we _could_ make the setup_db callable into another
    # service, and then we could ask it. Leaving that running forever when it's only
    # got one thing to do might seem excessive, but if there's a service to add new
    # images (and their embeddings) to the database, then we could use that to add
    # the capability of adding more images to _this_ app.
    # For now, we'll just ignore the problem for the moment...

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
