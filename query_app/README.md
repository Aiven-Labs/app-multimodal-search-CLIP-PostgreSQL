# A FastAPI app to retrieve some pictures that match a text prompt

This application has two parts:

When it starts up, it creates a background thread to

1. Wait for the CLIP app to be ready
2. Set up the database (enabling pgvector and creating the database table)
3. Calculate the embedding for each of the sample images (named in the
   `image_name.txt` file) and add a record for that image and its embedding
   into the database table. If this process has already been done for the
   database, it won't repeat it.

Meanwhile it also starts up the actual query frontend, which

1. Gets the text prompt from the user
2. Asks the CLIP app for the embedding for that text prompt
3. Queries the database for matches
4. Shows the first/best four matching pictures to the user

That will only work if the backgroud thread has completed. If not, you'll get
an informative message explaining how far it has got, and a request to try
again later.

## Prerequisites

### A PostgreSQL® database

You need an existing PostgreSQL® database, with the pgvector extension installed.

An Aiven for PostgreSQL service will do very well - see the
[Create a service](https://aiven.io/docs/products/postgresql/get-started#create-a-service)
section in the [Aiven documentation](https://aiven.io/docs).

### The CLIP application

The CLIP application is needed to generate embeddings, both for the sample
images, and also for the text prompts from the user.

To get the clip app running, see the [`clip_app` README](../clip_app/README.md)

## Set the environment variable to access your database

You need to tell the query app how to connect to the PostgreSQL database.
The URL you use should look something like
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

Run the app using fastapi
```shell
fastapi dev clip_app.py --port 3000
```

## Running with Docker

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

## Make a query

Go to http://127.0.0.1:3000 in a web browser, and request a search.

Possible ideas include:
* cat
* man jumping
* outer space

You should get four images back.

## The `find_images` script

If the CLIP app is running and the database is populated with examples, you can
run `find_images.py` to check that everything is working without
starting up the web app. It looks for images matching the text `man jumping` and
reports their filenames. It needs the same environment variables setting as the
main `clip_app`.
```shell
./find_images.py
```
