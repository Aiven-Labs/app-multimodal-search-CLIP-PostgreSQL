FROM python:3.11-slim

WORKDIR /app

COPY ./requirements.txt /app

# We need git to get the CLIP library, and 'slim' doesn't include it
# We also want curl
RUN apt-get update \
&& apt-get install -y --no-install-recommends git curl \
&& apt-get purge -y --auto-remove \
&& rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY ./app.py /app

RUN mkdir -p /app/templates
COPY ./templates/index.html  /app/templates/index.html
COPY ./templates/images.html /app/templates/images.html

RUN mkdir -p /app/models
RUN curl https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt --output /app/models/ViT-B-32.pt


EXPOSE 3000
CMD [ "fastapi", "run", "app.py", "--port", "3000" ]
