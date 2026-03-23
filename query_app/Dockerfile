FROM python:3.11-slim

WORKDIR /app

COPY ./pyproject.toml /app

RUN python3 -m pip install --no-cache-dir .

COPY ./query_app.py /app
COPY ./create_table.py /app
COPY ./process_images.py /app
COPY ./image_names.txt /app

RUN mkdir -p /app/templates
COPY ./templates/index.html  /app/templates/index.html
COPY ./templates/images.html /app/templates/images.html

# Create the database table, and add embeddings for some images
# (Ideally this would be a whole different application, and it would
# notify _this_ application when it had finished. This would mean we
# could get the query interface up quickly, and have this running in
# parallel, and the user would get some sort of "still populating
# the database" response if they tried asking something too early.)

#RUN /app/create_table.py
#RUN /app/process_images.py

EXPOSE 3000
CMD [ "fastapi", "run", "query_app.py", "--port", "3000" ]
