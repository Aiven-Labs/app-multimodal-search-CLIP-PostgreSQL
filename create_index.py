#!/usr/bin/env python3

"""Notebook 1: create the index.
"""

import os

from dotenv import load_dotenv

load_dotenv()
SERVICE_URI = os.getenv("SERVICE_URI")

from opensearchpy import OpenSearch
opensearch = OpenSearch(SERVICE_URI, use_ssl=True)

# Notebook 1: Create an index that supports knn vectors

index_name = 'photos'
index_body = {
  'settings': {
    'index': {
      "knn": True
    }
  },
  "mappings": {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 512
                }
            }
        }
}

try:
    opensearch.indices.create(index_name, body=index_body)
except Exception as exc:
    print(f'{exc.__class__.__name__}: {exc}')
