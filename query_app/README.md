# A FastAPI app to retrieve some pictures that match a text prompt

This application:

1. Gets the text prompt from the user
2. Asks the CLIP app for the embedding for that text prompt
3. Queries the database for matches
4. Shows the first/best four matching pictures to the user

## Prerequisites

### A PostgreSQL® database

You need an existing PostgreSQL® database, and you need to populate it with 
image names/URLs and their corresponding embeddings.

An Aiven for PostgreSQL service will do very well - see the
[Create a service](https://aiven.io/docs/products/postgresql/get-started#create-a-service)
section in the [Aiven documentation](https://aiven.io/docs).

### Sample data in the database

To populate the database with sample data, you need
1. To get the clip app running (you'll also need this to make queries). See 
   the [`clip_app` README](../clip_app/README.md)
2. To run the database setup script. This also depends on the clip app.
   See the [`setup_db` README](../setup_db/README.md).

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

You can run `find_images.py` to check that everything is working without
starting up the web app. It looks for images matching the text `man jumping` and
reports their filenames. It needs the same environment variables setting as the
main `clip_app`.
```shell
./find_images.py
```
