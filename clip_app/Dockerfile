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

# Bind to 0.0.0.0, not localhost or 127.0.0.1, so that we're visible
# to the other application
EXPOSE 8000
CMD [ "fastapi", "run", "clip_app.py", "--host", "0.0.0.0", "--port", "8000" ]
