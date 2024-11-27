#!/usr/bin/env python3

"""Notebook 2: process the images
"""

import os

import clip
import torch

from dotenv import load_dotenv
from opensearchpy import OpenSearch
from PIL import Image

load_dotenv()
SERVICE_URI = os.getenv("SERVICE_URI")

opensearch = OpenSearch(SERVICE_URI, use_ssl=True)

# Load the open CLIP model
print('Loading CLIP model')
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using {device}')
model, preprocess = clip.load("ViT-B/32", device=device)

index_name = "photos"  # Update with your index name

# Notebook 2: process and upload videos

# Path to the directory containing photos
image_dir = "photos"

# Batch size for processing images and indexing embeddings
batch_size = 100

from opensearchpy.helpers import bulk

def compute_clip_features(photos_batch):
    # Load all the photos from the files
    photos = [Image.open(photo_file) for photo_file in photos_batch]

    # Preprocess all photos
    photos_preprocessed = torch.stack([preprocess(photo) for photo in photos]).to(device)

    with torch.no_grad():
        # Encode the photos batch to compute the feature vectors and normalize them
        photos_features = model.encode_image(photos_preprocessed)
        photos_features /= photos_features.norm(dim=-1, keepdim=True)

    # Transfer the feature vectors back to the CPU and convert to numpy
    return photos_features.cpu().numpy()


def index_embeddings_to_opensearch(data):
    actions = []
    for d in data:
        action = {
            "_index": "photos",  # Update with your index name
            "_source": {
                "image_url": d['image_url'],
                "embedding": d['embedding'].tolist()
            }
        }
        actions.append(action)
    success, _ = bulk(opensearch, actions, index="photos")
    print(f"Indexed {success} embeddings to OpenSearch")


# Iterate over images and process them in batches

# List to store embeddings
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
        data.append({'image_url': file_path, 'embedding': embedding})

    # Check if we have enough data to index
    if len(data) >= batch_size:
        index_embeddings_to_opensearch(data)
        data = []

# Index any remaining data
if len(data) > 0:
    index_embeddings_to_opensearch(data)

print("All embeddings indexed successfully.")
