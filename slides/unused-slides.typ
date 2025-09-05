// Get Polylux from the official package repository
#import "@preview/polylux:0.4.0": *

// Fletcher for diagrams
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#import fletcher.shapes: pill, chevron

// Make the paper dimensions fit for a presentation and the text larger
#set page(paper: "presentation-16-9")

// Use big text for slides!
#set text(size: 25pt)

// And fonts. Typst seems to like Libertinus.
// (I do rather like the serif capital Q)
// I had to download it from the Releases folder at https://github.com/alerque/libertinus
// and install it using /Applications/Font Book
//
// Before that, I was being traditional and using "Times Roman" and "Helivitica"
// (or "Helvetica Neue")

#set text(font: "Libertinus Sans")

#show heading.where(
  level: 1
): it => text(
  font: "Libertinus Serif",
  it.body,
)

#show heading.where(
  level: 2
): it => text(
  font: "Libertinus Serif",
  it.body,
)

// UNUSED SLIDES

// Our images are in the GitHub repository, at
// https://github.com/Aiven-Labs/app-multimodal-search-CLIP-PostgreSQL/tree/main/photos
// but the files there are, well, files, so won't work in an `img` tag. Instead we need
// to refer to the raw content. This is OK for a demo, but should not be used in
// production, as GitHub is not really intended for this purpose!
// (and yes, this should not be hard coded, either)
#slide[
== Some constants
```python
index_name = "photos"  # Index name in PostgreSQL
image_dir = "photos"   # Path to the photos directory

PHOTOS_BASE = 'https://raw.githubusercontent.com/Aiven-Labs/app-multimodal-search-CLIP-PostgreSQL/refs/heads/main/photos/'

# Batch size for processing images and indexing embeddings
batch_size = 100
```
]




#slide[
== Process images in batches (1)
```python
# Iterate over images and process them in batches
data = []
image_files = os.listdir(image_dir)
for i in range(0, len(image_files), batch_size):
    print(f'Batch {i}')
    batch_files = image_files[i:i+batch_size]
    batch_urls = [f'{PHOTOS_BASE}/{file}'
                      for file in batch_files]
```
]

#slide[
== Process images in batches (2)

Compute embeddings for the batch of images and create data dictionary for indexing
```python
    batch_embeddings = compute_clip_features(
        batch_file_paths)

    for file_name, file_url, embedding \
      in zip(batch_files, batch_urls, batch_embeddings):
        data.append((file_name, file_url,
                     vector_to_string(embedding)))
```
]

#slide[
== Process images in batches (3)

```python
    # Check if we have enough data to index
    if len(data) >= batch_size:
        index_embeddings_to_postgres(data)
        data = []

# Don't forget: Index any remaining data
if len(data) > 0:
    index_embeddings_to_postgres(data)
```
]


#slide[
  == The application

  _Some_ of which is very similar to what we saw already
]

#slide[
== Some lovely imports
```python
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
```
]

#slide[
== Fastapi stuff
```python
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
```
]

#slide[
== Logging is good

Your web app _really_ wants logging
```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s:'
           ' %(message)s',
)

logger = logging.getLogger(__name__)
```
]

// Why didn't I just use DATABASE_URL as the constant? I don't remember...
// Shouldn't we have some error checking in case that constant ends up unset?
// (yes, yes we should)
#slide[
== Where's that database?
```python
SERVICE_URI = os.getenv("DATABASE_URL")
if not SERVICE_URI:
    # Try the .env file
    load_dotenv()
    SERVICE_URI = os.getenv("DATABASE_URL")
# At which point we rather hope we found the database URL
```
]

#slide[
== Our GET method
```python
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "search_hint": "Find images like...",
        },
    )
```
]

#slide[
== Our POST method (1)
```python
@app.post("/search_form", response_class=HTMLResponse)
async def search_form(
    request: Request,
    search_text: Annotated[str, Form()]):

    logging.info(f'Search form requests {search_text!r}')
```
]

#slide[
== Our POST method (2) - no CLIP model yet
```python
    if not clip_model.model:
        return templates.TemplateResponse(
            request=request,
            name="images.html",
            context={
                "images": [],
                "error_message": clip_model.error_string,
            }
        )
```
]

#slide[
== Our POST method (3) - do the search
```python
    results = search_for_matches(search_text)
    return templates.TemplateResponse(
        request=request,
        name="images.html",
        context={
            "images": results,
            "error_message": "",
        }
    )
```
]
