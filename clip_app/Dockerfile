FROM python:3.11-slim

WORKDIR /app

COPY ./pyproject.toml /app

RUN python3 -m pip install --no-cache-dir .

COPY clip_app.py /app
COPY ./model_info.py /app
COPY ./download_model.py /app

# Get the model so that it's ready when the app starts
RUN mkdir -p /app/models/
RUN /app/download_model.py

EXPOSE 8000
CMD [ "fastapi", "run", "clip_app.py", "--port", "8000" ]
