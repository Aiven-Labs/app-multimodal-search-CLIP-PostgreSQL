#!/usr/bin/env python3

"""Enable pgvector and create an appropriate table
"""

import os

import psycopg
from dotenv import load_dotenv

load_dotenv()
SERVICE_URI = os.getenv("PG_SERVICE_URI")

try:
    with psycopg.connect(SERVICE_URI) as conn:
        with conn.cursor() as cur:
            cur.execute('CREATE EXTENSION vector;')
            cur.execute('CREATE TABLE pictures (filename text PRIMARY KEY, embedding vector(512));')
except Exception as exc:
    print(f'{exc.__class__.__name__}: {exc}')
