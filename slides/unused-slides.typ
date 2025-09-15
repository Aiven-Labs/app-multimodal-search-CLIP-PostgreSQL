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


#slide[

  == Why vectors?

  Broadly, we can describe the characteristics of things with numbers.

  For instance, we can describe colours with RGB values,\ or hotels with ratings.
]

// Do I want to show a second arrow, or a second diagram with two arrows
// and the vector between them?
#slide[
  == RGB encoding as a 3d vector: `#e6cdb3`

  #import "@preview/cetz:0.4.1"

  #align(center)[
    #cetz.canvas(
      length: 15pt,
      background: luma(240),
      {
        import cetz.draw: *
        set-style(mark: (end: ">"))

        line((0, 0, 0), (10, 0, 0), name: "blue", stroke: black)
        line((0, 0, 0), (0, 10, 0), name: "red")
        line((0, 0, 0), (0, 0, 10), name: "green")

        content((12, 0, 0), text(fill: red)[ff,0,0])
        content((0, 11, 0), text(fill: green)[0,ff,0])
        content((1, 0, 12), text(fill: blue)[0,0,ff])

        line((9, 0, 0), (9, 0, 7), mark: (end: none), stroke: green)
        line((0, 0, 7), (9, 0, 7), mark: (end: none), stroke: green)

        line((9, 0, 0), (9, 8, 0), mark: (end: none), stroke: green)
        line((0, 8, 0), (9, 8, 0), mark: (end: none), stroke: green)

        line((0, 0, 7), (0, 8, 7), mark: (end: none), stroke: green)
        line((0, 8, 0), (0, 8, 7), mark: (end: none), stroke: green)

        line((9, 8, 0), (9, 8, 7), mark: (end: none), stroke: green)
        line((0, 8, 7), (9, 8, 7), mark: (end: none), stroke: green)
        line((9, 0, 7), (9, 8, 7), mark: (end: none), stroke: green)

        line((0, 0, 0), (9, 8, 7), stroke: (thickness: 5pt))

        content((0.5, -1, 7), [b3])
        content((-1, 8.5, 0), [cd])
        content((9.5, -1, 0), [e6])

        content((14, 9.5, 10), box(
          fill: rgb(90%, 80%, 70%),
          outset: 7pt,
          radius: 5pt,
        )[e6,cd,b3])
      },
    )
  ]
]

#slide[

  == We can do mathematics with vectors

  We can compare their

  - length
  - direction

  and we can do maths between vectors - for instance:

  #grid(
    columns: (auto, auto, auto),
    rows: (auto, auto),
    column-gutter: 5pt,
    row-gutter: 20pt,
    [Is #h(10pt)],
    highlight(fill: luma(230))[the vector between colour 1 and colour 2],
    [#h(10pt) _similar to_],

    [ ],
    [#highlight(fill: luma(230))[the vector between colour 3 and colour 4]],
    [?],
  )

  /*
  #quote(block: true)[
    Is #highlight(fill: luma(230))[the vector between colour 1 and colour 2]
    _similar to_
    #highlight(fill: luma(230))[the vector between colour 3 and colour 4]?
  ]
  */
]

#slide[
  == How do we calculate the vectors?
]

#slide[

  == Calculating the vectors by hand: early NLP

  In early Natural Language Processing, words would be categorised by hand.

  #align(center)[
    #grid(
      columns: (auto, auto, auto),
      gutter: 10pt,
      align: left,
      [`king`], [#sym.arrow.r.stroked], [`[1.0, 1.0, 0.8, ...]`],
      [`queen`], [#sym.arrow.r.stroked], [`[1.0, 0.0, 0.7, ...]`],
      [`princess`], [#sym.arrow.r.stroked], [`[0.9, 0.0, 0.3, ...]`],
    )
  ]

  gauging "importance", "gender", "typical age" and then other things

  This doesn't scale well - but we do know what the "meanings" are, and we can
  hope to spot bias
]

#slide[

  == Calculating the vectors using ML

  With ML, we can

  - *train* a machine learning system
  - to *"recognise"* that a thing belongs to particular categories.

  And the "thing" can be more than just words

  This is wonderful - but sometimes leads to surprising results, because we
  don't know what the meanings *"chosen"* actually are
]

// Olena's grid of emojis
#slide[
  == Classifying emoji
  #align(center)[
    #grid(
      columns: (auto, auto, auto),
      align: horizon,
      table(
        align: right,
        columns: (1.5em, 1.5em, 1.5em, 1.5em, 1.5em, 1.5em, 1.5em),
        rows: (1.5em, 1.5em, 1.5em, 1.5em, 1.5em, 1.5em, 1.5em),
        [ ], [0], [1], [2], [3], [4], [5],
        [0], [👗], [ ], [ ], [ ], [ ], [🐶],
        [1], [👠], [👟], [ ], [ ], [🐻], [ ],
        [2], [ ], [ ], [ ], [ ], [ ], [ ],
        [3], [💳], [💶], [ ], [ ], [ ], [ ],
        [4], [ ], [ ], [ ], [ ], [ ], [ ],
        [4], [ ], [ ], [ ], [ ], [ ], [🎸],
      ),
      grid.cell(inset: 0.5em, sym.arrow),
      table(
        align: right,
        columns: (1.5em, 1.5em, 1.5em),
        rows: (1.3em, 1.3em, 1.3em, 1.3em, 1.3em, 1.3em, 1.3em, 1.3em),
        [ ], [X], [Y],
        [👗], [0], [0],
        [👠], [0], [1],
        [👟], [1], [1],
        [🐶], [5], [0],
        [🐻], [4], [1],
        [💳], [0], [3],
        [💶], [1], [3],
        [🎸], [5], [5],
      ),
    )
  ]
]

// TRY NOT USING THIS SLIDE
// If I understand this correctly, this is actually doing the opposite of
// what we're doing - it's taking an image and returning a corresponding text
// or description
// Do I actually want/need this slide, or is the previous one sufficient?
#slide[
  == Training CLIP: Create dataset classifier, predict...

  #figure(
    image("images/CLIP-training-overview-b.svg", width: 14em),
    caption: text(size: 20pt)[source: https://github.com/openai/CLIP],
  )
]


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
