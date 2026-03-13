FROM python:3.11-slim

WORKDIR /app

COPY ./pyproject.toml /app

RUN python3 -m pip install --no-cache-dir .

COPY ./app.py /app
COPY ./model_info.py /app
COPY ./download_model.py /app

RUN mkdir -p /app/templates
COPY ./templates/index.html  /app/templates/index.html
COPY ./templates/images.html /app/templates/images.html

# Get the model so that it's ready when the app starts
RUN mkdir -p /app/models/
RUN /app/download_model.py


EXPOSE 3000
CMD [ "fastapi", "run", "app.py", "--port", "3000" ]
