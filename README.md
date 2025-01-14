# An app for searching for images matching a text, using CLIP, PostgreSQL® and pgvector

A Python web web app that searches for images matching a given text


First, create a virtual environment to keep package installation local to this directory
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
python3 -m pip install -r requirements.txt
```

> **Note** Sometimes we've seen the Python `clip.load` function fail to download the CLIP model, presumably due to the source server being busy. The code here will use a local copy of the model if it's available. To make that local copy:
>
>     mkdir models
>     curl https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt --output models/ViT-B-32.pt

Create your PostgreSQL® database. An Aiven for PostgreSQL service will do very
well - see the [Create a
service](https://aiven.io/docs/products/postgresql/get-started#create-a-service)
section in the [Aiven documentation](https://aiven.io/docs).

Copy the template environment file
```shell
cp .env_example .env
```
Then edit the `.env` file to insert the credentials needed to connect to the
database.

> **Note** If you're using an Aiven for PostgreSQL service, then you want the
**Service URI** value from the service **Overview** in the Aiven console.
> The result should look something like:
>
>     DATABASE_URL=postgres://<user>:<password>@<host>:<port>/defaultdb?sslmode=require

Enable pgvector and set up the table we need in the database
```shell
./create_table.py
```

Calculate the embeddings for the pictures in the `photos` directory, and
upload them to the database
```shell
./process_images.py
```

You can run `find_images.py` to check that everything is working - it looks
for images matching the text `man jumping` and reports their filenames
```shell
./find_images.py
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
