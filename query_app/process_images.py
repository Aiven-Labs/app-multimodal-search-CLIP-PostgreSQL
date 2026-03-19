#!/usr/bin/env python3

"""Calculate embeddings for our images, and upload them to PostgreSQL
"""

import logging
import os
import sys

from pathlib import Path

import httpx
import psycopg
import psycopg.errors

from dotenv import load_dotenv


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
if not DATABASE_URL:
    import sys
    sys.exit('No value found for environment variable DATABASE_URL (the PG database)')

# Get our model name
MODEL_NAME = os.environ.get('MODEL_NAME', 'openai/clip-vit-base-patch32')

# Get the URL for our CLIP embedding service
CLIP_SERVICE_URL = os.environ.get('CLIP_SERVICE_URL', 'http://localhost:8000')


index_name = "photos"  # Update with your index name

# Path to the directory containing photos
image_dir = Path("./photos").resolve()

# Our images are in the GitHub repository, at
# https://github.com/Aiven-Labs/app-multimodal-search-CLIP-PostgreSQL/tree/main/photos
# but the files there are not meant to be accessed as HTTP resources, so we need
# to refer to the raw content. This is OK for a demo, but should not be used in
# production, as GitHub is not really intended for this purpose!
# (and yes, this should not be hard coded, either)
PHOTOS_URL_BASE = 'https://raw.githubusercontent.com/Aiven-Labs/app-multimodal-search-CLIP-PostgreSQL/refs/heads/main/photos'

# Batch size for processing images and indexing embeddings
batch_size = 100


def compute_clip_features(photo_file_path: str) -> list[float]:
    try:
        response = httpx.post(
            f'{CLIP_SERVICE_URL}/embed',
            json={
                "model_name": MODEL_NAME,
                "datatype": "image",
                "value": photo_file_path,
            },
        )
        response.raise_for_status()

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
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                with cur.copy('COPY pictures (filename, url, embedding) FROM STDIN') as copy:
                    for row in data:
                        copy.write_row(row)
    except Exception as exc:
        logger.error(f'{exc.__class__.__name__}: {exc}')
        raise


def data_already_exists(file_name: str) -> bool:
    """Make a quick check to see if we've already got a record for this file.
    """
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT filename FROM pictures WHERE filename = %s;",
                    (file_name,),
                )
                results = cur.fetchall()
                return len(results) > 0
    except Exception as exc:
        logger.error(f'Unable to query database {exc.__class__.__name__}: {exc}')
        raise Exception(f'Unable to query database')


def vector_to_string(embedding: list[float]) -> str:
    """Convert our (ndarry) embedding vector into a string that SQL can use.
    """
    vector_str = ", ".join(str(x) for x in embedding)
    vector_str = f'[{vector_str}]'
    return vector_str


def main():
    # Iterate over images and process them in batches

    # Ideally we'd upload photos via an endpoint on the query app (which would
    # also allow us to give users a page where they could enter image URLs to
    # upload). For the moment, we're just going to use a pre-generated list of
    # the photos in the `photos` directory in the GitHub repository, the same
    # directory that PHOTOS_URL_BASE references, and do it all by hand.

    # The image_names.txt file is in the same directory as this file...
    running_dir = Path(__file__).parent.resolve()
    image_names_file = running_dir / 'image_names.txt'
    with open(image_names_file) as fd:
        image_file_names = fd.read().splitlines()

    # If the data is already in the database, then we don't want to run again
    # So let's look for the _last_ filename
    if data_already_exists(image_file_names[-1]):
        logger.info("Data is already in the database")
        return

    data = []

    # Process images in batches
    for i in range(0, len(image_file_names), batch_size):
        logger.info(f'Batch {i}')
        batch_file_names = image_file_names[i:i+batch_size]
        batch_file_urls = [f'{PHOTOS_URL_BASE}/{name}' for name in batch_file_names]

        # Create data dictionary for indexing
        for file_name, file_url in zip(batch_file_names, batch_file_urls):
            embedding = compute_clip_features(file_url)
            data.append((file_name, file_url, vector_to_string(embedding)))

        # Check if we have enough data to index
        if len(data) >= batch_size:
            index_embeddings_to_postgres(data)
            data = []

    # Index any remaining data
    if len(data) > 0:
        logger.info('Remaining embeddings')
        index_embeddings_to_postgres(data)

    logger.info("All embeddings indexed successfully.")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('CTRL-C\n')
        sys.exit(0)
    except Exception as exc:
        # We should already have logged whatever went wrong, but let's make sure
        print(exc)
        sys.exit(1)
