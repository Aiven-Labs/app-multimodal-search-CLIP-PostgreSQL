#!/usr/bin/env python3

"""An app to find (the first four) images matching a text string, and display them.
"""

import asyncio
import logging
import os
import time

from contextlib import asynccontextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Annotated, Callable, List, Sequence, Union
from urllib.parse import urlparse

import httpx
import psycopg
import torch

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form
from fastapi import HTTPException
from fastapi import status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from PIL.ImageFile import ImageFile
from pydantic import BaseModel
from transformers import CLIPProcessor, CLIPModel

# Get our model name and directories
from download_model import download_model
from model_info import *

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

# Our images are in the GitHub repository, at
# https://github.com/Aiven-Labs/app-multimodal-search-CLIP-PostgreSQL/tree/main/photos
# but the files there are not meant to be accessed as HTTP resources, so we need
# to refer to the raw content. This is OK for a demo, but should not be used in
# production, as GitHub is not really intended for this purpose!
# (and yes, this should not be hard coded, either)
PHOTOS_URL_BASE = 'https://raw.githubusercontent.com/Aiven-Labs/app-multimodal-search-CLIP-PostgreSQL/refs/heads/main/photos'

# Batch size for processing images and indexing embeddings
batch_size = 100

# Let's keep a readiness status
class AppStatus:
    ready = False
    message = 'Loading necessary information'

app_status = AppStatus()

# ===========================================================================
# Download the CLIP model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Model:
    """The types of the values are found from the docstring for clip.load

    See also the source code at https://github.com/openai/CLIP/blob/main/clip/clip.py

    (we could just make them type Any, but it's interesting to know the actual types)
    """
    model: Union[None, torch.nn.Module]
    preprocess: Union[None, Callable[[Image], torch.Tensor]]

clip_model = Model(None, None)


def load_clip_model():
    """Load the open CLIP model"""

    # If the MODEL_DIR doesn't exist, then assume we need to download the model.
    # We *could* just allow the call of `CLIPModel.from_pretrained` to do the
    # download for us, but our `download_model` function actually downloads
    # less data / fewer files, so should be a bit quicker and use less space.
    if not MODEL_DIR.exists():
        download_model()

    try:
        # Load the open CLIP model that we just downloaded
        logger.info(f'Importing CLIP model {MODEL_NAME} from {MODEL_DIR}')
        logger.info(f'Using device {DEVICE} for model calculations')
        clip_model.model = CLIPModel.from_pretrained(MODEL_DIR).to(DEVICE)
        clip_model.processor = CLIPProcessor.from_pretrained(MODEL_DIR)
        logger.info(f'CLIP model {MODEL_NAME} imported')
    except Exception as exc:
        logger.exception(f'Unable to load CLIP model {MODEL_NAME}')


# ===========================================================================
# SET UP CLIP MODEL AND DATABASE

def create_table():
    """Enable pgvector and set up our table.
    """
    # Enable pgvector seperately, in case I DROP the table and want to recreate it
    logger.info('Enabling pgvector')
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute('CREATE EXTENSION IF NOT EXISTS vector;')
    except psycopg.OperationalError as exc:
        logger.exception(f'Error talking to database (enabling pgvector); {exc.__class__.__name__}: {exc}')
        raise Exception(f"Error connecting to database: {exc}")
    except Exception as exc:
        logger.exception(f'Error enabling pgvector; {exc.__class__.__name__}: {exc}')
        raise Exception(f"Error enabling pgvector: {exc}")

    logger.info('Creating table')
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f'CREATE TABLE {TABLE_NAME} (filename text PRIMARY KEY, url text, embedding vector(512));',
                )
    except psycopg.errors.DuplicateTable as exc:
        # The table already existed
        logger.info(f'{exc.__class__.__name__}: {exc}')
    except Exception as exc:
        logger.exception(f'Error creating table {TABLE_NAME}; {exc.__class__.__name__}: {exc}')
        raise Exception(f"Error creating table {TABLE_NAME}: {exc}")


def index_embeddings_to_postgres(data):
    """Write a batch of data rows to PostgreSQL

    It's probably a bit wasteful to create a new connection for each batch,
    but it means we don't need to worry about a potentially long running
    connection.

    See https://www.psycopg.org/psycopg3/docs/basic/copy.html for more on
    the use of COPY.
    """
    logger.info(f'Writing {len(data)} rows to PostgreSQL')
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                with cur.copy(f'COPY {TABLE_NAME} (filename, url, embedding) FROM STDIN') as copy:
                    for row in data:
                        copy.write_row(row)
    except Exception as exc:
        logger.exception(f'{exc.__class__.__name__}: {exc}')
        raise Exception("Unable to write to database")


def entry_already_exists(file_name: str) -> bool:
    """Make a quick check to see if we've already got a record for this file.
    """
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f'SELECT filename FROM {TABLE_NAME} WHERE filename = %s;',
                    (file_name,),
                )
                results = cur.fetchall()
                return len(results) > 0
    except Exception as exc:
        logger.exception(f'Unable to query database {exc.__class__.__name__}: {exc}')
        raise Exception(f'Unable to query database')


def populate_table():
    logger.info('Adding image embeddings to the database')
    # Iterate over images and process them in batches

    # For the moment, we're going to prepare a pre-generated list of
    # the photos in the `photos` directory in the GitHub repository, the same
    # directory that PHOTOS_URL_BASE references, and do it all by hand.

    # The image_names.txt file is in the same directory as this file...
    running_dir = Path(__file__).parent.resolve()
    image_names_file = running_dir / 'image_names.txt'
    try:
        with open(image_names_file) as fd:
            image_file_names = fd.read().splitlines()
    except FileNotFoundError:
        logger.error(f'Cannot open {image_names_file}')
        # Best we can do is give up loading things and hope there's something in the database
        return

    # If the data is already in the database, then we don't want to run again
    # So let's look for the _last_ filename
    if entry_already_exists(image_file_names[-1]):
        logger.info("Data is already in the database")
        return

    # NOTE that this whole process is not robust if the database is not yet set up
    # and multiple instances of the app are running, all trying to update the database :(

    # Especially during development, this app can be run after the database table has
    # already been partially populated. We don't want the cost of getting an embedding
    # for an entry we've already got, so we'll put up with extra SQL queries, assuming
    # that's faster/cheaper.

    # Work out which files we need to add and which are there already
    logger.info(f'Checking {len(image_file_names)} filenames')
    app_status.message = 'Checking examples in PostgreSQL database. Please try again later'
    # We assume that we always use the same list of file names, in the same order.
    # Given that, check until we find a missing entry. Since we already checked
    # if the last entry was present (above) we expect not to have to check them all :)

    # Our "null hypothesis" is if there are no entries, then we use the same list
    actual_file_names = image_file_names

    for index, filename in enumerate(image_file_names):
        # If we already have an entry in the database, skip it
        if not entry_already_exists(filename):
            logger.info(f'Entry {index} is not there - we can add the rest')
            actual_file_names = image_file_names[index:]
            break

    logger.info(f'Adding {len(actual_file_names)} image embeddings to the database in batches of {batch_size}')
    total = 0
    data = []
    app_status.message = 'Adding examples to PostgreSQL database. Please try again later'

    for i in range(0, len(actual_file_names), batch_size):
        batch_files = actual_file_names[i:i + batch_size]
        batch_urls = [f'{PHOTOS_URL_BASE}/{file}' for file in batch_files]

        batch_image_data = [get_image_data(url) for url in batch_urls]

        # Compute embeddings for the batch of images
        batch_embeddings = get_image_embeddings(batch_image_data)

        # Create data dictionary for indexing
        for file_name, file_url, embedding in zip(batch_files, batch_urls, batch_embeddings):
            data.append((file_name, file_url, vector_to_string(embedding)))

        # Check if we have enough data to index
        if len(data) >= batch_size:
            index_embeddings_to_postgres(data)
            total += len(data)
            data = []
            app_status.message = f'Adding examples to PostgreSQL database ({total}/{len(actual_file_names)}). Please try again later'

    # Index any remaining data
    if data:
        logger.info(f'Adding remaining {len(data)} image embeddings')
        index_embeddings_to_postgres(data)

    logger.info("All image embeddings added")


# ===========================================================================
# LIFECYCLE

# This gets run as a background task when the app starts up.
# That means the UI is available as soon as possible, and if the background task
# has not yet completed, the user gets suitable information when they try a prompt
def setup_clip_and_database():
    app_status.message = "Loading CLIP model. Please try again later"
    load_clip_model()
    if not clip_model:
        app_status.message = "Unable to load CLIP model. Please restart the app"
        # And we give up - `app_status.ready` will never be set True
        return

    try:
        app_status.message = "PostgreSQL database is not set up. Please try again later"
        create_table()
        app_status.message = "Adding examples to PostgreSQL database. Please try again later"
        populate_table()
    except Exception as e:
        logger.exception(f'Unable to setup database {e.__class__.__name__}: {e}', stack_info=True)
        app_status.message = f'Unable to setup database: {e}. Please fix and then restart app'
        # And we give up - `app_status.ready` will never be set True
        return

    app_status.ready = True
    app_status.message = "App is ready for queries"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Define events at the start and end of the app lifespan

    This means we can start the query app before the CLIP service ready,
    and still give _some_ response to the user
    """
    logger.info('Async load task starting')
    blocking_loader = asyncio.to_thread(setup_clip_and_database)
    background_task = asyncio.create_task(blocking_loader)
    yield
    # We don't have an unload step

# ===========================================================================
# QUERIES

def get_image_data(url: str) -> ImageFile:
    """Load image data from a URL.

    We assume a "file:" URL for a local file, or an "http:" or "https:" URL
    for remote data.

    May raise an Exception if an error occurs retrieving the remote data
    """
    if url.startswith('file:'):
        parsed_url = urlparse(url)
        file_path = Path(parsed_url.path)
        return Image.open(file_path)

    # Retrieve the URL
    try:
        response = httpx.get(
            url,
            follow_redirects=True,  # For instance, we know that the GitHub URLs we use will redirect
        )
        response.raise_for_status()
    except Exception as exc:
        raise Exception(f'Error getting image {url}: {exc}')

    # Turn the bytes into a "file like" object for PIL
    image_bytes = BytesIO(response.content)

    return Image.open(image_bytes)


def get_image_embeddings(image_data: Sequence[ImageFile]):
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

    # Return the feature vectors
    return features.numpy()


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


def vector_to_string(embedding):
    """Convert our (ndarry) embedding vector into a string that SQL can use.
    """
    vector_str = ", ".join(str(x) for x in embedding)
    vector_str = f'[{vector_str}]'
    return vector_str


def search_for_matches(text):
    """Returns pairs of the form (image_name, image_url)"""
    logger.info(f'Searching for {text!r}')
    vector = get_text_embedding(text)

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
        #raise Exception(f'Error queryi1Gng database: {exc}')


app = FastAPI(redirect_slashes=False, lifespan=lifespan)
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

    if not app_status.ready:
        logger.info(f'App status: {app_status.message}')
        return templates.TemplateResponse(
            request=request,
            name="images.html",
            context={
                "images": [],
                "error_message": f"{app_status.message}",
            }
        )

    try:
        results = search_for_matches(search_text)
    except Exception as e:
        logger.exception(f'Error searching for {search_text!r}: {e.__class__.__name__}: {e}', stack_info=True)
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
