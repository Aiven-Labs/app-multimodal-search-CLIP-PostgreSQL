# Search for images matching a text, using CLIP, PostgreSQL® and pgvector


A Python web app that searches for images matching a given text

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

The instructions for the first two are below.

A summary of how to do the last two
[is also below](#running-individual-services), but details are in the 
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

And when that's all running1G, go to http://0.0.0.0:3000/ to find the prompt.


## Running individual services

The order in which things are done matters, because the different services 
depend on each other.

1. Create an external database, as described in [One service and an external 
   database, using compose](#one-service-and-an-external-database-using-compose)
2. Start the CLIP application, as described in
   [`the clip_app README`](./clip_app/README.md)
3. Run the database setup script, as described in 
   [`the setup_db README`](./setup_db/README.md)
4. Start the query application, as described in
   [the `query_app README`](./query_app/README.md)

And when that's all running, go to http://0.0.0.0:3000/ to find the prompt.

## Other considerations

### The sample photos

The images in the `photos` directory are the same as those used in [Workshop: Searching for images with vector search - OpenSearch and CLIP model](https://github.com/Aiven-Labs/workshop-multimodal-search-CLIP-OpenSearch).

They came from Unsplash and have been reduced in size to make them fit within
GitHub filesize limits for a repository.

> **Note** Both `setup_db` and `query_app` retrieve the sample images
> directly from this GitHub repository. This is not good practice for a
> production app, as GitHub is not intended to act as an image repository for
> web apps.

### Use the right Python image

When writing the Dockerfile, the default `FROM python:3.11` downloads much
of Ubuntu, which we don't need. We can vastly reduce the size of the image
by using `FROM python:3.11-slim`, at the cost of needing to install `git`
(needed by the requirements to download
`git+https://github.com/openai/CLIP.git`) and `curl`. See
https://hub.docker.com/_/python for more about the Python images available.

### Use `redirect_slashes=FALSE` in FastAPI

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
