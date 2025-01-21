FROM python:3.11-slim

WORKDIR /app

COPY ./requirements.txt /app

# We need git to get the CLIP library, and 'slim' doesn't include it
RUN apt-get update \
&& apt-get install -y --no-install-recommends git \
&& apt-get purge -y --auto-remove \
&& rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY ./app.py /app

RUN mkdir -p /app/templates
COPY ./templates/index.html  /app/templates/index.html
COPY ./templates/images.html /app/templates/images.html

EXPOSE 3000
CMD [ "fastapi", "run", "app.py", "--port", "3000" ]
