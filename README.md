# An app for searching for images matching a text, using CLIP, PostgreSQL and pgvector

A Python web web app that searches for images matching a given text


```shell
python3 -m venv venv
source venv/bin/activate
```

```shell
python3 -m pip install -r requirements.txt
```

> ![TIP] NSometimes we've seen `clip.load` function fail to download the CLIP model, presumably due to the source server being busy. The code here will use a local copy of the model if it's available. To make that local copy:
>
>     mkdir models
>     curl https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt --output models/ViT-B-32.pt

```shell
cp .env_example .env
```

Edit the `.env` file to insert the credentials needed to connect to the database

Enable pgvector and set up the table we need in the database
```shell
./create_table.py
```

Calculate the embeddings for the pictures in the `photos` directory, and
upload them to the database
```shell
./process_images
```

You can run `find_images.py` to check that everything is working - it looks
for images matching `man jumping` and reports their filenames
```shell
./find_images
```

Run the webapp locally using fastapi
```shell
fastapi dev app.py
```

Go to http://127.0.0.1:8000 in a web browser, and request a search.

Possible ideas include:
* cat
* man jumping
* outer space

## The photos

The images in the `photos` directory are the same as those used in [Workshop: Searching for images with vector search - OpenSearch and CLIP model](https://github.com/Aiven-Labs/workshop-multimodal-search-CLIP-OpenSearch).

They came from Unsplash and have been reduced in size to make them fit within
GitHub filesize limits for a repository.


## Inspirations

* The [Workshop: Searching for images with vector search - OpenSearch and CLIP
  model](https://github.com/Aiven-Labs/workshop-multimodal-search-CLIP-OpenSearch)
  which does (essentially) the same thing, but using OpenSearch and Jupyter
  notebooks.

* [Building a movie recommendation system with Tensorflow and
  PGVector](https://github.com/Aiven-Labs/pgvector-tensorflow-movie-recommendations-workshop)
  which searches text, and produces a web app using JavaScript

For help understanding how to use HTMX
* [Using HTMX with FastAPI](https://testdriven.io/blog/fastapi-htmx/)
* and for help understanding how I wanted to use forms, [Updating Other Content
](https://htmx.org/examples/update-other-content/) from the HTMX documentation
(I went for option 1, as suggested).
