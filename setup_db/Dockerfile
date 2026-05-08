FROM python:3.11-slim

WORKDIR /app

COPY ./pyproject.toml /app

RUN python3 -m pip install --no-cache-dir .

COPY ./image_names.txt /app
COPY ./setup_db.py /app

# Create the database table, and add embeddings for some images
# Assumes the database already exists.
# Waits for the CLIP service to be ready to serve embeddings before it
# starts its work.

CMD [ "python3", "setup_db.py"]
