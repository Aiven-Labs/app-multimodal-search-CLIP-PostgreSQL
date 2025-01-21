# An app for searching for images matching a text, using CLIP, PostgreSQL® and pgvector

A Python web web app that searches for images matching a given text

## Prepare the database

> **Note** you need to do this stage whether you're going to run the app
> locally or via Docker.

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

Create your PostgreSQL® database. An Aiven for PostgreSQL service will do very
well - see the [Create a
service](https://aiven.io/docs/products/postgresql/get-started#create-a-service)
section in the [Aiven documentation](https://aiven.io/docs).

Either:

* Copy the template environment file
  ```shell
  cp .env_example .env
  ```
  Then edit the `.env` file to insert the credentials needed to connect to the database.

  > **Note** If you're using an Aiven for PostgreSQL service, then you want the
  **Service URI** value from the service **Overview** in the Aiven console.
  > The result should look something like:
  >
  >     DATABASE_URL=postgres://<user>:<password>@<host>:<port>/defaultdb?sslmode=require
  
or:

* Set the `DATABASE_URL` environment variable to the PostgreSQL Service URI.

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


## Running the app locally

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
python3 -m pip install -r requirements.txt
```

> **Note** Sometimes we've seen the Python `clip.load` function fail to
> download the CLIP model, presumably due to the source server being busy. The
> code here will use a local copy of the model if it's available. To make that
> local copy:
>
>     mkdir models
>     curl https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt --output models/ViT-B-32.pt

If you set up a `.env` file containing the PostgreSQL service URI, then you're
ready to run the app. Otherwise, you can set a local environmment variable,
`DATABASE_URL`, to that same URI.

Run the webapp locally using fastapi
```shell
fastapi dev app.py
```

Go to http://127.0.0.1:8000 in a web browser, and request a search.

Possible ideas include:
* cat
* man jumping
* outer space

## Running with Docker

Build the image.
```
docker build -t appimage .
```

Run the container. Pass the PostgreSQL service URI as an environment variable.
```
docker run -d --name mycontainer -p 3000:3000 -e DATABASE_URL=$DATABASE_URL appimage
```

Then go to http://localhost:3000/ and the app should be running.

## The photos

The images in the `photos` directory are the same as those used in [Workshop: Searching for images with vector search - OpenSearch and CLIP model](https://github.com/Aiven-Labs/workshop-multimodal-search-CLIP-OpenSearch).

They came from Unsplash and have been reduced in size to make them fit within
GitHub filesize limits for a repository.

> **Note** When the app is running, it retrieves the images for the HTML page
> directly from this GitHub repository. This is not good practice for a
> production app, as GitHub is not intended to act as an image repository for
> web apps.

## Interesting things I learnt

1. When writing the Dockerfile, the default `FROM python:3.11` downloads much
   of Ubuntu, which we don't need. We can vastly reduce the size of the image
   by using `FROM python:3.11-slim`, at the cost of needing to install `git`
   (needed by the requirements to download
   `git+https://github.com/openai/CLIP.git`) and `curl`. See
   https://hub.docker.com/_/python for more about the Python images available.

2. In the Docker context, we deliberately download the CLIP image file
   ```
   RUN curl https://openaipublic.azureedge.net/clip/models/.../ViT-B-32.pt --output /app/models/ViT-B-32.pt
   ```
   so that the file is ready for the app when it runs. This makes a lot of sense
   in a container context.

3. At one point I was running the Dockerised application in an HTTPS context.
   In order to make the HTMX redirect to `/search_form` also use HTTPS, I
   needed to tell FastAPI `redirect_slashes=FALSE` (and make sure that the
   `/search_form` in the `templates/index.html` file didn't end with `/`).

   I found the information at [FastAPI redirection for trailing slash returns
   non-SSL
   link](https://stackoverflow.com/questions/63511413/fastapi-redirection-for-trailing-slash-returns-non-ssl-link)
   very helpful, particularly [this
   comment](https://stackoverflow.com/questions/63511413/fastapi-redirection-for-trailing-slash-returns-non-ssl-link#:~:text=Since%20FastAPI%20version%200.98.0%20the%20framework%20provides%20a%20way%20to%20disable%20the%20redirect%20behaviour%20by%20setting%20the%20redirect_slashes%20parameter%20to%20False%2C%20which%20is%20True%20by%20default.%20This%20works%20for%20the%20whole%20application%20as%20well%20as%20for%20individual%20routers.).


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
