# Search for images matching a text, using CLIP, PostgreSQL® and pgvector


A Python web app that searches for images matching a given text

> **Note:** The [slides/](slides/) directory contains slides for a 25
> minute talk about [version 1](https://github.com/Aiven-Labs/app-multimodal-search-CLIP-PostgreSQL/tree/v1.0.0) 
> of this code, as given at PyCon UK 2025. That version was organised around 
> a single app and its container file.

## Architecture

There are two components in use here:

* A PostgreSQL® database, with the `pgvector` extension installed. This 
  is used to store image and text embeddings
* A FastAPI application that can take a text, or the URL for am image file, 
  and use the CLIP model to calculate the vector embedding for that text or 
  image.

When the app starts up, it runs a background task to:

1. Download the CLIP model.
2. Make sure that `pgvector` is enabled in the database
3. Calculate the embedding for each of the sample images (named in the
   `image_name.txt` file) and add a record for that image and its embedding
   into the database table. If this process has already been done for the
   database, it won't repeat it.

   > **Note:** It doesn't attempt to cope with two instances of this app both 
   > trying to populate the database at the same time.
 
Meanwhile it also starts the actual query frontend, which

1. Gets a text prompt from the user
2. Asks the CLIP model for the embedding for that text prompt
3. Queries the database for matches
4. Shows the first/best four matching pictures to the user

That will only work if the backgroud thread has completed. If not, you'll get
an informative message explaining how far it has got, and a request to try
again later.

![Showing the first match for "man jumping" in the query app](slides/images/app-man-jumping.png)

## Four ways to run this code

1. With its own PostgreSQL database, using the `compose.yaml` file.
2. With an external PostgreSQL database, using the `compose,existing-db.yaml`
3. At the command line, using an external PG database.
4. Via the container file, using an external PG database.

<details>
<summary>**Using compose to create all the services, including PostgreSQL**</summary>

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
</details>

<details>
<summary>Using an external PostgreSQL database</summary>

### Create your external PostgreSQL® database

Remember that the PostgreSQL database needs to have the pgvector extension
installed.

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
</details>

<details>
<summary>Using compose with an external PostgreSQL database</summary>

Set up the external database - see
[Using an external PostgreSQL database](#using-an-external-postgresql-database)
above.

### Create the images and start the services

```shell
docker compose -f compose.existing-db.yaml up -d
```

And when that's all running, go to http://0.0.0.0:3000/ to find the prompt.
</details>

<details>
<summary>At the command line, using an external PG database</summary>

Set up the external database - see
[Using an external PostgreSQL database](#using-an-external-postgresql-database)
above.

Change into the `query_app` directory
```shell
cd query_app
```

If you didn't already do so, create a virtual environment to keep
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

Run the app using fastapi
```shell
fastapi dev query_app.py --port 3000
```
</details>

<details>
<summary>Via the container file, using an external PG database</summary>

Set up the external database - see
[Using an external PostgreSQL database](#using-an-external-postgresql-database)
above.

Change into the `query_app` directory
```shell
cd query_app
```

Build the image.
```
docker build -t query_app_image .
```

Run the container. Pass the PostgreSQL service URI as an environment variable.
```
docker run -d --name query_app_container \
    -p 3000:3000 \
    -e DATABASE_URL=$DATABASE_URL \
    query_app_image
```
</details>

## Make a query

Go to http://127.0.0.1:3000 in a web browser, and request a search.

Possible ideas include:
* cat
* man jumping
* outer space

You should get four images back.


## Other considerations

### The sample photos

The images in the `photos` directory are the same as those used in [Workshop: Searching for images with vector search - OpenSearch and CLIP model](https://github.com/Aiven-Labs/workshop-multimodal-search-CLIP-OpenSearch).

They came from Unsplash and have been reduced in size to make them fit within
GitHub filesize limits for a repository.

> **Note:** The `query_app` retrieves the sample images
> directly from this GitHub repository. This is not good practice for a
> production app, as GitHub is not intended to act as an image repository for
> web apps. We **could** copy the `photos` directory into the container image,
> at the cost of about 43MB. That would also speed up populating the database,
> as local files would be read.

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
