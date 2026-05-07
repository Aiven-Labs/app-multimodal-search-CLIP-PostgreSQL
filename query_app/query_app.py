#!/usr/bin/env python3

"""An app to find (the first four) images matching a text string, and display them.
"""

import asyncio
import logging
import os
import time

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, List

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

# Let's keep a readiness status
class AppStatus:
    ready = False
    message = 'Waiting for CLIP service'

app_status = AppStatus()

# ===========================================================================
# SET UP CLIP MODEL AND DATABASE

# Our images are in the GitHub repository, at
# https://github.com/Aiven-Labs/app-multimodal-search-CLIP-PostgreSQL/tree/main/photos
# but the files there are not meant to be accessed as HTTP resources, so we need
# to refer to the raw content. This is OK for a demo, but should not be used in
# production, as GitHub is not really intended for this purpose!
# (and yes, this should not be hard coded, either)
PHOTOS_URL_BASE = 'https://raw.githubusercontent.com/Aiven-Labs/app-multimodal-search-CLIP-PostgreSQL/refs/heads/main/photos'

# Batch size for processing images and indexing embeddings
batch_size = 100

TIME_TO_WAIT_FOR_CLIP_SERVICE = 20


def wait_for_clip_service() -> None:
    """Wait for the CLIP service to be ready.
    """
    logger.info('Waiting for CLIP service')
    while True:
        try:
            response = httpx.get(
                f'{CLIP_SERVICE_URL}/healthy',
                timeout=None,    # the default is documented as 5 seconds
            )
            if response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
                # Wait for a moment...
                time.sleep(TIME_TO_WAIT_FOR_CLIP_SERVICE)
                continue
            else:
                response.raise_for_status()

            logger.info('CLIP service is ready')
            return

        except Exception as exc:
            logger.error(f'Error getting CLIP service readiness from {CLIP_SERVICE_URL}: {exc.__class__.__name__}: {exc}')
            raise Exception(f'Error getting CLIP service readiness from {CLIP_SERVICE_URL}: {exc.__class__.__name__}: {exc}')


def create_table():
    """Enable pgvector and set up our table.

    Assumes that if anything goes wrong, it's something we can ignore
    """
    # Enable pgvector seperately, in case I DROP the table and want to recreate it
    logger.info('Enabling pgvector')
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute('CREATE EXTENSION IF NOT EXISTS vector;')
    except Exception as exc:
        logger.error(f'Error enabling pgvector; {exc.__class__.__name__}: {exc}')
        raise

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
        logger.error(f'Error creating table {TABLE_NAME}; {exc.__class__.__name__}: {exc}')
        raise


def compute_clip_features(photo_file_path: str) -> list[float]:
    #logger.info(f'Requesting embeddings for {photo_file_path}')
    try:
        response = httpx.post(
            f'{CLIP_SERVICE_URL}/embed',
            json={
                "model_name": MODEL_NAME,
                "datatype": "image",
                "value": photo_file_path,
            },
            timeout=None,    # the default is documented as 5 seconds
        )
        response.raise_for_status()

        #logger.info(f'Received embeddings for {photo_file_path}')

        data = response.json()
        return data["embedding"]
    except Exception as exc:
        logger.error(f'Error getting image embeddings from {CLIP_SERVICE_URL}: {exc.__class__.__name__}: {exc}')
        #raise Exception('Unable to get text embedding')
        raise Exception(f'Error getting image embeddings from {CLIP_SERVICE_URL}: {exc.__class__.__name__}: {exc}')


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
        logger.error(f'{exc.__class__.__name__}: {exc}')
        raise


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
        logger.error(f'Unable to query database {exc.__class__.__name__}: {exc}')
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
    # for an entry we've already got, so we'll put up with an extra SQL query, assuming
    # that's faster/cheaper.
    logger.info(f'Adding {len(image_file_names)} image embeddings to the database in batches of {batch_size}')
    data = []
    logged_skipping = False
    batch_count = 1
    total = 0
    for filename in image_file_names:
        # If we already have an entry in the database, skip it
        if entry_already_exists(filename):
            if not logged_skipping:
                logger.info(f'Skipping entries that already exist')
                logged_skipping = True
            total += 1
            continue

        # Calculate the embedding for this filename's data and add it to our list
        file_url = f'{PHOTOS_URL_BASE}/{filename}'
        embedding = compute_clip_features(file_url)
        data.append((filename, file_url, vector_to_string(embedding)))

        if len(data) >= batch_size:
            logger.info(f'Adding batch {batch_count} of image embeddings')
            index_embeddings_to_postgres(data)
            batch_count += 1
            total += len(data)
            data = []
            app_status.message = f'Adding examples to PostgreSQL database ({total}/{len(image_file_names)})'

    # Index any remaining data
    if data:
        logger.info(f'Adding remaining {len(data)} image embeddings')
        index_embeddings_to_postgres(data)

    logger.info("All image embeddings added")


def setup_database():
    app_status.message = "Waiting for CLIP service"
    wait_for_clip_service()
    app_status.message = "PostgreSQL database is not set up"
    create_table()
    app_status.message = "Adding examples to PostgreSQL database"
    populate_table()
    app_status.ready = True
    app_status.message = "App is ready for queries"


# ===========================================================================
# LIFECYCLE

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Define events at the start and end of the app lifespan

    This means we can start the query app before the CLIP service ready,
    and still give _some_ response to the user
    """
    logger.info('Async load task starting')
    blocking_loader = asyncio.to_thread(setup_database)
    background_task = asyncio.create_task(blocking_loader)
    yield
    # We don't have an unload step


# ===========================================================================
# QUERIES

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
        return templates.TemplateResponse(
            request=request,
            name="images.html",
            context={
                "images": [],
                "error_message": f"{app_status.message} - please try again later",
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
