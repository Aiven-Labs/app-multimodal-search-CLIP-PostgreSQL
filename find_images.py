#!/usr/bin/env python3

"""Find images that match a (hard coded) text string and report their filenames.

This is intended as a way to check that everything necessary to run the app is
actually working
"""

import logging
import os

from pathlib import Path

import psycopg
import torch

from dotenv import load_dotenv
from transformers import CLIPProcessor, CLIPModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
)

logger = logging.getLogger(__name__)

load_dotenv()
SERVICE_URI = os.getenv("DATABASE_URL")


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


INDEX_NAME = "photos"  # Update with your index name


def get_single_embedding(text):
    with torch.no_grad():
        inputs = processor(
            text=[text],
            return_tensors='pt',
            padding=True,           # do we need this?
        ).to(DEVICE)

        # Compute the feature vectors
        features = model.get_text_features(**inputs)

        # Normalise the embeddings, to make them easier to compare
        features /= features.norm(dim=-1, keepdim=True)

    # Return the feature vector (there is just the one)
    return features.numpy()[0]


def vector_to_string(embedding):
    """Convert our (ndarry) embedding vector into a string that SQL can use.
    """
    vector_str = ", ".join(str(x) for x in embedding.tolist())
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
    vector = get_single_embedding(text)

    embedding_string = vector_to_string(vector)

    # Perform search
    try:
        with psycopg.connect(SERVICE_URI) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM pictures ORDER BY embedding <-> %s LIMIT 4;",
                    (embedding_string,),
                )
                rows = cur.fetchall()
                return [row[0] for row in rows]
    except Exception as exc:
        print(f'{exc.__class__.__name__}: {exc}')
        return []


text_input = "man jumping"  # Provide your text input here
logger.info(f'Searching for {text_input!r}')
matches = search_for_matches(text_input)

for index, filename in enumerate(matches):
    print(f'{index+1}: {filename}')
