#!/usr/bin/env python3

"""Enable pgvector and create an appropriate table
"""

import os

import psycopg
from dotenv import load_dotenv

SERVICE_URI = os.getenv("DATABASE_URL")
if not SERVICE_URI:
    # Try the .env file
    load_dotenv()
    SERVICE_URI = os.getenv("DATABASE_URL")
if not SERVICE_URI:
    import sys
    sys.exit('No value found for environment variable DATABASE_URL (the PG database)')

# Enable pgvector seperately, in case I DROP the table and want to recreate it
print('Enabling pgvector')
try:
    with psycopg.connect(SERVICE_URI) as conn:
        with conn.cursor() as cur:
            cur.execute('CREATE EXTENSION vector;')
except Exception as exc:
    print(f'{exc.__class__.__name__}: {exc}')

print('Creating table')
try:
    with psycopg.connect(SERVICE_URI) as conn:
        with conn.cursor() as cur:
            cur.execute('CREATE TABLE pictures (filename text PRIMARY KEY, url text, embedding vector(512));')
except Exception as exc:
    print(f'{exc.__class__.__name__}: {exc}')

print('Done')
