# Set up the database ready for use

This script

1. Creates the database table we're using, if it doesn't already exist
2. Calculate embeddings for our sample image files and enters them into
   the database table, if they're not already there.

## Prerequisites

### A PostgreSQL® database

You need an existing PostgreSQL® database, which this script will
populate with image names/URLs and their corresponding embeddings.

> **Note** an Aiven for PostgreSQL database will work just fine.

### The clip app

To calculate the embeddings for the sample images, you will need the clip 
app running. See the [`clip_app` README](../clip_app/README.md)

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
./setup_db.py
```

## Running with Docker

Build the image.
```
docker build -t setup_db_image .
```

Run the container. Pass the PostgreSQL service URI as an environment variable.
```
docker run -d --name setup_db_container \
    -e DATABASE_URL=$DATABASE_URL \
    setup_db_image
```
