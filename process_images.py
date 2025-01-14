#!/usr/bin/env python3

"""Calculate embeddings for our images, and upload them to PostgreSQL
"""

import os

from pathlib import Path

import clip
import psycopg
import torch

from dotenv import load_dotenv
from PIL import Image


SERVICE_URI = os.getenv("DATABASE_URL")
if not SERVICE_URI:
    # Try the .env file
    load_dotenv()
    SERVICE_URI = os.getenv("DATABASE_URL")
if not SERVICE_URI:
    import sys
    sys.exit('No value found for environment variable DATABASE_URL (the PG database)')

# Load the open CLIP model
# If we download it remotely, it will default to being cached in ~/.cache/clip
LOCAL_MODEL = Path('./models/ViT-B-32.pt').absolute()
MODEL_NAME = 'ViT-B/32'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if LOCAL_MODEL.exists():
    print(f'Importing CLIP model {MODEL_NAME} from {LOCAL_MODEL.parent}')
    print(f'Using {DEVICE}')
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE, download_root=LOCAL_MODEL.parent)
else:
    print(f'Importing CLIP model {MODEL_NAME}')
    print(f'Using {DEVICE}')
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)

index_name = "photos"  # Update with your index name

# Notebook 2: process and upload videos

# Path to the directory containing photos
image_dir = "photos"

# Batch size for processing images and indexing embeddings
batch_size = 100


def compute_clip_features(photos_batch):
    # Load all the photos from the files
    photos = [Image.open(photo_file) for photo_file in photos_batch]

    # Preprocess all photos
    photos_preprocessed = torch.stack([preprocess(photo) for photo in photos]).to(DEVICE)

    with torch.no_grad():
        # Encode the photos batch to compute the feature vectors and normalize them
        photos_features = model.encode_image(photos_preprocessed)
        photos_features /= photos_features.norm(dim=-1, keepdim=True)

    # Transfer the feature vectors back to the CPU and convert to numpy
    return photos_features.cpu().numpy()


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
                with cur.copy('COPY pictures (filename, embedding) FROM STDIN') as copy:
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
    batch_file_paths = [os.path.join(image_dir, file) for file in batch_files]

    # Compute embeddings for the batch of images
    batch_embeddings = compute_clip_features(batch_file_paths)

    # Create data dictionary for indexing
    for file_path, embedding in zip(batch_file_paths, batch_embeddings):
        data.append((file_path, vector_to_string(embedding)))

    # Check if we have enough data to index
    if len(data) >= batch_size:
        index_embeddings_to_postgres(data)
        data = []

# Index any remaining data
if len(data) > 0:
    print('Remaining embeddings')
    index_embeddings_to_postgres(data)

print("All embeddings indexed successfully.")
