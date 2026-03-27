# A FastAPI app to query a (local) CLIP model

This application

1. Gets the CLIP model
2. Answers requests for embeddings produced using that model

## Running the app in the shell

First, if you didn't already do so, create a virtual environment to keep
package installation local to this directory
```shell
python3 -m venv venv
```

Enable it - this shows doing so for a normal Unix shell, there are other
scripts for (for instance) the `fish` shell
```shell
source venv/bin/activate
```

Install the Python packages we need
```shell
python3 -m pip install .
```

If you want, you can download a local copy of the model to `./models/` with
```shell
./model_download.py
```
This means that the app will use that copy, otherwise it will look for a
cached version (for instance, in `~/.cache/huggingface`) and if there isn't
one, will download the model itself, which will cause a delay before the app
can find images.

Run the app using fastapi
```shell
fastapi dev clip_app.py --port 8000
```

## Running with Docker

Build the image.
```
docker build -t clip_app_image .
```

Run the container.
```
docker run -d --name clip_app_container -p 8000:8000 clip_app_image
```

## Endpoints

### POST `/embed`

Make a POST to the `embed` endpoint to get an embedding.

It takes a JSON request looking like:
```json
{
  "model_name": "openai/clip-vit-base-patch32",
  "datatype": "text",
  "value": "Man jumping"
}
```

- Only the one model name is supported at the moment.
- The `datatype` may be either `text`, in which case the `value` should be a 
  text string, or `image`, in which case the `value` should be the URL for 
  an image file.
- There is limited support for `file://` image URLs - basically the code 
  just removes the leading `file:` and any `/` characters and then looks for 
  the image file locally.

It returns a response of the form
```json
{"embedding":[0.014406335540115833,..,0.007464071735739708]}
```
where the `..` is the appropriate number of floating point values.

### GET '/started'

This always responds 200 `{"status": "ok"}`, indicating that the app has 
started.

### GET `/healthy`

This responds with whether the app is ready to provide embeddings. It 
returns status 200 or 503:

* 200 `{"status": "ok"}`, if it is ready to provide embeddings.
* 503 with content like `{"detail": CLIP model openai/clip-vit-base-patch32 not
  loaded yet - try again soon"}` if it is not yet ready.


## Examples

Getting the embedding for a text:
```shell
curl -X 'POST' \
  'http://127.0.0.1:8000/embed' \
  -i \
  -H 'Content-Type: application/json' \
  -d '{
    "model_name": "openai/clip-vit-base-patch32",
    "datatype": "text",
    "value": "Man jumping"
  }'
```

Getting the embedding for an image:
```shell
curl -i -X 'POST' \
  'http://127.0.0.1:8000/embed' \
  -H 'Content-Type: application/json' \
  -d '{
    "model_name": "openai/clip-vit-base-patch32",
    "datatype": "image",
    "value": "https://raw.githubusercontent.com/Aiven-Labs/app-multimodal-search-CLIP-PostgreSQL/refs/heads/main/photos/718nqUl20Mo.jpg"
  }'
```

## History - which CLIP package

The original version of this code used the Python CLIP library from
https://github.com/openai/CLIP. Unfortunately, that has a `setup.py` that
requires `pkg_resources`, which is removed in Python 3.12 and setuptools 82.

Rather than try to cope with legacy code issues, the code here has been
changed to use HuggingFace transformers instead, still with the same CLIP
model (HuggingFace call it `openai/clip-vit-base-patch32` instead of
`ViT-B/32`). As a benefit, this should also make it easier to change the code
to use different models.

## Only download the model files we need

In the Docker context, we deliberately download the model first,
so that the app starts in a more deterministic state (there isn't a
possibly long wait while the Python script loads the model from the
internet).

However, we don't need *all* of the files within a HuggingFace model,
and ignoring some of them can save a useful amount of space (and time).

See `model_info.py` and `download_model.py` for more information.
