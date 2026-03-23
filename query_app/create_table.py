#!/usr/bin/env python3

"""Enable pgvector and create an appropriate table
"""

import logging
import os

import psycopg
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
)

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    # Try the .env file
    load_dotenv()
    DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    import sys
    logger.error('No value found for environment variable DATABASE_URL (the PG database)')
    sys.exit(1)

# Enable pgvector seperately, in case I DROP the table and want to recreate it
logger.info('Enabling pgvector')
try:
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute('CREATE EXTENSION vector;')
except Exception as exc:
    logger.error(f'{exc.__class__.__name__}: {exc}')

logger.info('Creating table')
try:
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute('CREATE TABLE pictures (filename text PRIMARY KEY, url text, embedding vector(512));')
except Exception as exc:
    logger.error(f'{exc.__class__.__name__}: {exc}')

logger.info('Done')
