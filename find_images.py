#!/usr/bin/env python3

"""Notebook 3: find things
"""

import logging
import os

import clip
import torch

from dotenv import load_dotenv
from opensearchpy import OpenSearch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
)

logger = logging.getLogger(__name__)

load_dotenv()
SERVICE_URI = os.getenv("SERVICE_URI")

logger.info('Creating OpenSearch connection')
opensearch = OpenSearch(SERVICE_URI, use_ssl=True)

# Load the open CLIP model
logger.info('Importing CLIP model')
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f'Using {device}')
model, preprocess = clip.load("ViT-B/32", device=device)

INDEX_NAME = "photos"  # Update with your index name


def get_single_embedding(text):
    with torch.no_grad():
        # Encode the text to compute the feature vector and normalize it
        text_input = clip.tokenize([text]).to(device)
        text_features = model.encode_text(text_input)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # Return the feature vector
    return text_features.cpu().numpy()[0]


def knn_search(text):
    vector = get_single_embedding(text)

    body = {
        "query": {
            "knn": {
                "embedding": {
                    "vector": vector.tolist(),  # Convert to list
                    "k": 4  # Number of nearest neighbors to retrieve
                }
            }
        }
    }

    # Perform search
    result = opensearch.search(index=INDEX_NAME, body=body)
    return result


def find_images(result):
    # Check if hits are present in the result
    if 'hits' in result and 'hits' in result['hits']:
        hits = result['hits']['hits']

        # Loop through each hit, up to a maximum of 4
        for i, hit in enumerate(hits[:4]):
            if '_source' in hit and 'image_url' in hit['_source']:
                image_url = hit['_source']['image_url']

                print(f"Found image {i+1}: {image_url}")
            else:
                print(f"Hit {i+1} does not contain an 'image_url' key.")

    else:
        print("Invalid result format or no hits found.")


text_input = "man jumping"  # Provide your text input here
logger.info(f'Searching for {text_input!r}')
result = knn_search(text_input)

find_images(result)
