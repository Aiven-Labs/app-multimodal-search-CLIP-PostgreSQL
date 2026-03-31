FROM python:3.11-slim

WORKDIR /app

COPY ./pyproject.toml /app

RUN python3 -m pip install --no-cache-dir .

COPY ./image_names.txt /app
COPY ./setup_db.py /app

# Create the database table, and add embeddings for some images
# (Ideally this would be a whole different application, and it would
# notify _this_ application when it had finished. This would mean we
# could get the query interface up quickly, and have this running in
# parallel, and the user would get some sort of "still populating
# the database" response if they tried asking something too early.)

CMD [ "python3", "setup_db.py"]
