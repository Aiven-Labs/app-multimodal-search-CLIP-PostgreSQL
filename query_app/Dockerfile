FROM python:3.11-slim

WORKDIR /app

COPY ./pyproject.toml /app

RUN python3 -m pip install --no-cache-dir .

COPY ./image_names.txt /app
COPY ./model_info.py /app
COPY ./download_model.py /app
COPY ./query_app.py /app

# Create the directory where we'll download the model
RUN mkdir -p /app/models/

RUN mkdir -p /app/templates
COPY ./templates/index.html  /app/templates/index.html
COPY ./templates/images.html /app/templates/images.html

EXPOSE 3000
CMD [ "fastapi", "run", "query_app.py", "--port", "3000" ]
