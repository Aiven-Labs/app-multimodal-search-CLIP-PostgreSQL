#!/usr/bin/env python3

"""Calculate embeddings for our images, and upload them to PostgreSQL
"""

import os

from pathlib import Path

import psycopg
import torch

from dotenv import load_dotenv
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


SERVICE_URI = os.getenv("DATABASE_URL")
if not SERVICE_URI:
    # Try the .env file
    load_dotenv()
    SERVICE_URI = os.getenv("DATABASE_URL")
if not SERVICE_URI:
    import sys
    sys.exit('No value found for environment variable DATABASE_URL (the PG database)')


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using device {DEVICE} for model calculations')

# Load the open CLIP model
# If we download it remotely, it will default to being cached in ~/.cache/clip
MODEL_NAME = 'openai/clip-vit-base-patch32'
LOCAL_MODEL = Path(f'./models/{MODEL_NAME}').absolute()
if LOCAL_MODEL.exists():
    print(f'Importing CLIP model {MODEL_NAME} from {LOCAL_MODEL.parent}')
    model = CLIPModel.from_pretrained(pretrained_model_name_or_path=LOCAL_MODEL).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(pretrained_model_name_or_path=LOCAL_MODEL)
else:
    print(f'Importing CLIP model {MODEL_NAME}')
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

index_name = "photos"  # Update with your index name

# Notebook 2: process and upload videos

# Path to the directory containing photos
image_dir = "photos"

# Our images are in the GitHub repository, at
# https://github.com/Aiven-Labs/app-multimodal-search-CLIP-PostgreSQL/tree/main/photos
# but the files there are, well, files, so won't work in an `img` tag. Instead we need
# to refer to the raw content. This is OK for a demo, but should not be used in
# production, as GitHub is not really intended for this purpose!
# (and yes, this should not be hard coded, either)
PHOTOS_BASE = 'https://raw.githubusercontent.com/Aiven-Labs/app-multimodal-search-CLIP-PostgreSQL/refs/heads/main/photos/'

# Batch size for processing images and indexing embeddings
batch_size = 100


def compute_clip_features(photos_batch):
    # Load all the photos from the files
    photos = [Image.open(photo_file) for photo_file in photos_batch]

    with torch.no_grad():  # We don't need gradients for inference, so can save memory
        # Preprocess all photos - resize and normalise
        inputs = processor(
            images=photos,
            return_tensors='pt',
            padding=True,           # do we need this?
        ).to(DEVICE)

        # Compute the feature vectors
        features = model.get_image_features(**inputs)

        # Normalise the embeddings, to make them easier to compare
        features /= features.norm(dim=-1, keepdim=True)


    # Convert the feature vectors back to numpy
    return features.numpy()


def index_embeddings_to_postgres(data):
    """Write a batch of data rows to PostgreSQL

    It's probably a bit wasteful to create a new connection for each batch,
    but it means we don't need to worry about a potentially long running
    connection.

    See https://www.psycopg.org/psycopg3/docs/basic/copy.html for more on
    the use of COPY.
    """
    try:
        with psycopg.connect(SERVICE_URI) as conn:
            with conn.cursor() as cur:
                with cur.copy('COPY pictures (filename, url, embedding) FROM STDIN') as copy:
                    for row in data:
                        copy.write_row(row)
    except Exception as exc:
        print(f'{exc.__class__.__name__}: {exc}')


def vector_to_string(embedding):
    """Convert our (ndarry) embedding vector into a string that SQL can use.
    """
    vector_str = ", ".join(str(x) for x in embedding.tolist())
    vector_str = f'[{vector_str}]'
    return vector_str


# Iterate over images and process them in batches

data = []

# Process images in batches
image_files = os.listdir(image_dir)
for i in range(0, len(image_files), batch_size):
    print(f'Batch {i}')
    batch_files = image_files[i:i+batch_size]
    batch_urls = [f'{PHOTOS_BASE}/{file}' for file in batch_files]

    # Compute embeddings for the batch of images
    batch_file_paths = [os.path.join(image_dir, file) for file in batch_files]
    batch_embeddings = compute_clip_features(batch_file_paths)

    # Create data dictionary for indexing
    for file_name, file_url, embedding in zip(batch_files, batch_urls, batch_embeddings):
        data.append((file_name, file_url, vector_to_string(embedding)))

    # Check if we have enough data to index
    if len(data) >= batch_size:
        index_embeddings_to_postgres(data)
        data = []

# Index any remaining data
if len(data) > 0:
    print('Remaining embeddings')
    index_embeddings_to_postgres(data)

print("All embeddings indexed successfully.")
