# An app for searching for images matching a text, using CLIP, PostgreSQL® and pgvector


A Python web wapp that searches for images matching a given text

> **Note:** The [slides/](slides/) directory contains slides for a 25
> minute talk about [version 1](https://github.com/Aiven-Labs/app-multimodal-search-CLIP-PostgreSQL/tree/v1.0.0) 
> of this code, as given at PyCon UK 2025. That version was organised around 
> a single app and its container file.

## Architecture

There are four components in use here:

* A PostgreSQL® database, with the `pgvector` extension installed. This 
  is used to store image and text embeddings
* A FastAPI application that can take a text, or the URL for am image file, 
  and use the CLIP model to calculate the vector embedding for that text or 
  image.
* A script that sets the database up. It makes sure that `pgvector` is set 
  up, and then uses the CLIP app to calculate the embedding for each image 
  in the `photos` directory. It adds an entry in the database for each image 
  name/URL and its embedding.
* A FastAPI application that allows the user to enter a text string. It uses 
  the CLIP app to calculate the embedding for the text string, and then 
  looks in the database for images with a similar embedding, so that it can 
  present the four closest images to the user.

![Showing the first match for "man jumping" in the query app](slides/images/app-man-jumping.png)

## Four ways to run this code

1. As a single self-contained service, complete with its own PostgreSQL 
   database, using the `compose.yaml` file.
2. As a single self-contained service with an external PostgreSQL database, 
   using the `compose-implicit-db.yaml`
3. As three separate services at the command line, using an external PG
   database.
4. As three separate containers, using an external PG database.

The instructions for the first two are on below.

A summary of how to do the last two is below, but details are in the 
individual README files in each service subdirectory
([`clip_app`](./clip_app/README.md),
[`setup_db`](./setup_db/README.md),
[`query_app`](./query_app/README.md)).

## One service using compose

### Set environment variables to describe your database

These will be used when creating the database service.

* For bash or other traditional shells:
  ```shell
  export POSTGRES_USER=embeddings_user
  export POSTGRES_PASSWORD=please-do-not-use-this-password
  export POSTGRES_DB=embeddings
  ```

* For the fish shell:
  ```shell
  set -x POSTGRES_USER embeddings_user
  set -x POSTGRES_PASSWORD please-do-not-use-this-password
  set -x POSTGRES_DB embeddings
  ```

* **Or** set the same values in a `.env` file
  ```shell
  POSTGRES_USER=embeddings_user
  POSTGRES_PASSWORD=please-do-not-use-this-password
  POSTGRES_DB=embeddings
  ```
  
> And as it says, please use a proper password 🙂.

### Create the images and start the services:

```shell
docker compose up -d
```

And when that's all running, go to http://0.0.0.0:3000/ to find the prompt.


## One service and an external database, using compose

### Create your external PostgreSQL® database

An Aiven for PostgreSQL service will do very well - see the
[Create a service](https://aiven.io/docs/products/postgresql/get-started#create-a-service)
section in the [Aiven documentation](https://aiven.io/docs).

### Set the environment variable to access your database

Since the database already exists, you need to let the other services know 
how to connect to it. The URL you need should look something like
> `postgres://<user>:<password>@<host>:<port>/dbname?sslmode=require`

We'll refer to that URL as `<service URI>` in the following notes.

> **Note** If you're using an Aiven for PostgreSQL service, then you can
> find this as the **Service URI** value from the service **Overview** in the
> Aiven console.

* For bash or other traditional shells:
  ```shell
  export DATABASE_URL=<service URI>
  ```

* For the fish shell:
  ```shell
  set -x DATABASE_URL=<service URI>
  ```

* **Or** set the same values in a `.env` file
  ```shell
  DATABASE_URL=<service URI>
  ```

### Create the images and start the services

```shell
docker compose -f compose-implicit-db.yaml up -d
```

And when that's all running, go to http://0.0.0.0:3000/ to find the prompt.


# Individual services

The order in which things are done matters, because the different services 
depend on each other.

1. Create an external database, as described in [One service and an external 
   database, using compose](#one-service-and-an-external-database-using-compose)
2. Start the CLIP application, as described in
   [`the clip_app README`](./clip_app/README.md)
3. Run the database setup script, as described in 
   [`the setup_db README`](./setup_db/README.md),
4. Start the query application, as described in
   [the `query_app README`](./query_app/README.md)).

And when that's all running, go to http://0.0.0.0:3000/ to find the prompt.

----

> **OLD TEXT BELOW HERE**

----

## Prepare the database

> **Note:** You need to do this stage whether you're going to run the app
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
python3 -m pip install .
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
python3 -m pip install .
```

If you set up a `.env` file containing the PostgreSQL service URI, then you're
ready to run the app. Otherwise, you can set a local environmment variable,
`DATABASE_URL`, to that same URI.

If you want, you can download a local copy of the model to `./models/` with
```shell
./model_download.py
```
This means that the app will use that copy, otherwise it will look for a
cached version (for instance, in `~/.cache/huggingface`) and if there isn't
one, will download the model itself, which will cause a delay before the app
can find images.

Run the webapp locally using fastapi
```shell
fastapi dev clip_app.py
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

## History - which CLIP package

The original version of this code used the Python CLIP library from
https://github.com/openai/CLIP. Unfortunately, that has a `setup.py` that
requires `pkg_resources`, which is removed in Python 3.12 and setuptools 82.

Rather than try to cope with legacy code issues, the code here has been
changed to use HuggingFace transformers instead, still with the same CLIP
model (HuggingFace call it `openai/clip-vit-base-patch32` instead of
`ViT-B/32`). As a benefit, this should also make it easier to change the code
to use different models.

## Interesting things I learnt

### Use the right distribution

When writing the Dockerfile, the default `FROM python:3.11` downloads much
of Ubuntu, which we don't need. We can vastly reduce the size of the image
by using `FROM python:3.11-slim`, at the cost of needing to install `git`
(needed by the requirements to download
`git+https://github.com/openai/CLIP.git`) and `curl`. See
https://hub.docker.com/_/python for more about the Python images available.

### Only download the model files we need

In the Docker context, we deliberately download the model first,
so that the app starts in a more deterministic state (there isn't a
possibly long wait while the Python script loads the model from the
internet).

However, we don't need *all* of the files within a HuggingFace model,
and ignoring some of them can save a useful amount of space (and time).

See `model_info.py` and `download_model.py` for more information.

### Using `redirect_slashes=FALSE` in FastAPI

At one point I was running the Dockerised application in an HTTPS context.
In order to make the redirect to `/search_form` also use HTTPS, I
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
  notebooks, and the OpenAI CLIP model.

* [Building a movie recommendation system with Tensorflow and
  PGVector](https://github.com/Aiven-Labs/pgvector-tensorflow-movie-recommendations-workshop)
  which searches text, and produces a web app using JavaScript.

For help understanding how to use HTMX
* [Using HTMX with FastAPI](https://testdriven.io/blog/fastapi-htmx/)
* and for help understanding how I wanted to use forms, [Updating Other Content
](https://htmx.org/examples/update-other-content/) from the HTMX documentation
(I went for option 1, as suggested).
