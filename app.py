#!/usr/bin/env python3

import logging
import os

from typing import Annotated, Union

import clip
import torch

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from opensearchpy import OpenSearch

app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.mount("/photos", StaticFiles(directory="photos"), name="photos")


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

index_name = "photos"  # Update with your index name


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
    result = opensearch.search(index=index_name, body=body)
    return result


def find_images(search_text):
    logger.info(f'Searching for {search_text!r}')
    result = knn_search(search_text)

    image_urls = []
    if 'hits' in result and 'hits' in result['hits']:
        hits = result['hits']['hits']

        # Loop through each hit, up to a maximum of 4
        for i, hit in enumerate(hits[:4]):
            if '_source' in hit and 'image_url' in hit['_source']:
                image_urls.append(hit['_source']['image_url'])
            else:
                logging.warning(f"Hit {i+1} does not contain an 'image_url' key.")

    else:
        logging.error("Invalid result format or no hits found.")

    return image_urls


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "search_hint": "Find images like...",
        },
        #context={"images": []},
    )


@app.post("/search_form", response_class=HTMLResponse)
async def search_form(request: Request, search_text: Annotated[str, Form()]):
    logging.info(f'Search form requests {search_text!r}')
    images = find_images(search_text)
    return templates.TemplateResponse(
        request=request,
        name="images.html",
        context={
            #"images": ['HQw3rB-2ON8.jpg', 'RIwKjm3TVMU.jpg'],
            "images": images,
        }
    )


@app.get("/search/{text}")
def read_item(text: str, q: Union[str, None] = None):
    logger.info(f'Searching for {text!r}')
    result = knn_search(text)

    image_urls = []
    if 'hits' in result and 'hits' in result['hits']:
        hits = result['hits']['hits']

        # Loop through each hit, up to a maximum of 4
        for i, hit in enumerate(hits[:4]):
            if '_source' in hit and 'image_url' in hit['_source']:
                image_urls.append(hit['_source']['image_url'])
            else:
                logging.warning(f"Hit {i+1} does not contain an 'image_url' key.")

    else:
        logging.error("Invalid result format or no hits found.")

    return {"image_urls": image_urls}
