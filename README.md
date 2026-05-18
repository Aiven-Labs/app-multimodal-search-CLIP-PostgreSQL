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

## Five ways to run this code

1. Using the compose file `compose.yaml` file, which creates its own PostgreSQL 
   database.
2. Using the compose file `compose.aiven.yaml` and deploying on
   [Aiven Apps](https://aiven.io/apps) - this runs the database and the query
   app within the Aiven platform
3. Using the compose file `compose,existing-db.yaml`, with an existing database
4. At the command line, with an existing database
5. Using the container file `query_app/Dockerfile`, with an external database

<details>
<summary>1. Using compose to create all the services, including 
PostgreSQL</summary>

## Using compose to create all the services, including PostgreSQL

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
<summary>2. Using Aiven Apps to deploy the services</summary>

## Using Aiven Apps to deploy the services

> **Note:** At the moment (May 2026) Aiven Apps is in Limited Availability (LA).
> See [Aiven Apps](https://aiven.io/apps) for more information and how to get
> access.

How to deploy Aiven Apps is described in the Aiven documentation at
[Deploy an  app](https://aiven.io/docs/products/apps/deploy-apps).

The following is a summary - check the documentation for the most up-to-date 
information.

1. Fork this repository. Connect your GitHub account to your Aiven organization.
2. In the [Aiven Console](https://console.aiven.io/) go to your project and 
   click **Applications**.
3. Click **Deploy app**.
4. Select your **Account**, your forked repository, and the `main` branch.
5. Click **Next**.
6. Select the manifest file `compose.aiven.yaml` and click **Scan**.
7. Change the configuration of the app components as needed: click the pen 
   icon on each card. By default a new PostgreSQL service will be created,
   but you can also choose to use an existing one.
8. To deploy the app services, click **Deploy**. 

The PostgreSQL service will start up, and the query app service will 
automatically connect to it.

The query app service **Overview** page will show the URL for the query page.

</details>

<details>
<summary>Setting up an external PostgreSQL database</summary>

## Setting up an external PostgreSQL database

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
<summary>3. Using compose with an existing PostgreSQL database</summary>

## Using compose with an existing PostgreSQL database

Set up the external database - see
[Setting up an external PostgreSQL database](#setting-up-an-external-postgresql-database)
above.

### Create the images and start the services

```shell
docker compose -f compose.existing-db.yaml up -d
```

And when that's all running, go to http://0.0.0.0:3000/ to find the prompt.
</details>

<details>
<summary>4. At the command line, with an existingPG database</summary>

## At the command line, with an existing PG database

Set up the external database - see
[Setting up an external PostgreSQL database](#setting-up-an-external-postgresql-database)
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
<summary>5. Using the container file, with an existing PG database</summary>

## Using the container file, with an existing PG database

Set up the external database - see
[Setting up an external PostgreSQL database](#setting-up-an-external-postgresql-database)
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
