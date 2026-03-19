#!/usr/bin/env python3

"""Find images that match a (hard coded) text string and report their filenames.

This is intended as a way to check that everything necessary to run the app is
actually working
"""

import logging
import os

import httpx
import psycopg

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
)

logger = logging.getLogger(__name__)

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# Get our model name
MODEL_NAME = os.environ.get('MODEL_NAME', 'openai/clip-vit-base-patch32')

# Get the URL for our CLIP embedding service
CLIP_SERVICE_URL = os.environ.get('CLIP_SERVICE_URL', 'http://localhost:8000')


def get_text_embedding(text) -> list[float]:
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
        raise Exception(f'Error getting text embedding from {CLIP_SERVICE_URL}: {exc.__class__.__name__}: {exc}')


def vector_to_string(embedding):
    """Convert our (ndarry) embedding vector into a string that SQL can use.
    """
    vector_str = ", ".join(str(x) for x in embedding)
    vector_str = f'[{vector_str}]'
    return vector_str


def search_for_matches(text):
    """Search for the "nearest" four images

    See [Querying](https://github.com/pgvector/pgvector?tab=readme-ov-file#querying)
    in the pgvector documentation.

    pgvector distance functions (see are:

    * <-> - L2 distance
    * <#> - (negative) inner product
    * <=> - cosine distance
    * <+> - L1 distance (added in 0.7.0)
    """
    vector = get_text_embedding(text)

    embedding_string = vector_to_string(vector)

    # Perform search
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM pictures ORDER BY embedding <-> %s LIMIT 4;",
                    (embedding_string,),
                )
                rows = cur.fetchall()
                return [row[0] for row in rows]
    except Exception as exc:
        logger.error(f'{exc.__class__.__name__}: {exc}')
        return []

def main():
    text_input = "man jumping"  # Provide your text input here
    logger.info(f'Searching for {text_input!r}')
    matches = search_for_matches(text_input)

    for index, filename in enumerate(matches):
        logger.info(f'{index+1}: {filename}')

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(exc)