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

// Trying out QR codes
#import "@preview/tiaoma:0.3.0"

// I want URLs to be visibly such - otherwise they'll just be shown as normal text
// #show link: underline
#show link: set text(blue)

// Footer
#let slide-footer = context[
  #set text(size: 15pt, fill:gray)
  #toolbox.side-by-side[
    #align(left)[#toolbox.slide-number / #toolbox.last-slide-number]
  ][
    #align(right)[Mastodon: \@much_of_a, Bluesky: \@tibs]
  ]
]
#set page(footer: slide-footer)


// If a quote has an attribution, but is not marked as "block", then write
// the attribution after it (in smaller text)
// Adapted from an example in the Typst documentation for quote's attribution
// parameter, at https://typst.app/docs/reference/model/quote/
#show quote.where(block: false): it => {
  ["] + h(0pt, weak: true) + it.body + h(0pt, weak: true) + ["]
  if it.attribution != none [ -- #text(size:20pt)[#it.attribution]]
}

// Give a very light grey background to code blocks
#show raw.where(block: true): it => block(
  fill: luma(240),
  inset: 5pt,
  it
)

// But be more subtle with inline code
#show raw.where(block: false): box.with(
  fill: luma(240),
  inset: (x: 3pt, y: 0pt),
  outset: (y: 3pt),
)


// Use #slide to create a slide and style it using your favourite Typst functions
#slide[
  #set align(horizon)

  // = Explaining the 5 types of database and how to choose between them

  #heading(
    level: 1,
    [Type text, find pictures: an app using CLIP, PostgreSQL® and pgvector]
  )

  #v(20pt)

  Tibs (they / he)

  #grid(
    columns: 2,

    text(size: 20pt)[

      21#super[st] September 2025, PyCon UK 2025

      Slides available at https://github.com/Aiven-Labs/app-multimodal-search-CLIP-PostgreSQL/slides
    ],

    align(right,
      grid(
        rows: (auto, auto),
        align: center,
        row-gutter: 10.0pt,
        tiaoma.qrcode("https://aiven.io/tibs", options: (scale: 3.0)),
        text(size: 20pt)[https://aiven.io/tibs]
      )
    )
  )
]

#slide[
  == Let's get started...

]

#slide[

  #grid(
    columns: (auto, auto, auto),
    align: horizon,
    table(
    align: right,
      columns: (1.5em, 1.5em, 1.5em, 1.5em, 1.5em, 1.5em),
      rows:  (1.5em, 1.5em, 1.5em, 1.5em, 1.5em, 1.5em),
      [0], [1], [2], [3], [4], [5],
      [1], [👗], [ ], [ ], [ ], [🐶],
      [2], [👠], [👟], [ ], [🐻], [ ],
      [3], [💳], [💶], [ ], [ ], [ ],
      [4], [ ], [ ], [ ], [ ], [ ],
      [5], [ ], [ ], [ ], [ ], [🎸],
    ),
    grid.cell(inset: 0.5em, sym.arrow),
    table(
    align: right,
      columns: (1.5em, 1.5em, 1.5em),
      rows: (1.5em, 1.5em, 1.5em, 1.5em, 1.5em, 1.5em, 1.5em, 1.5em),
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

// I really should create this slide using the data from the previous slide,
// rather than copying the whole thing!
// On the other hand, an I convinced this is a slide *I* can use to explain
// the concepts? Maybe not...
#slide[

  #grid(
    columns: (auto, auto, auto, auto),
    align: horizon,
    table(
    align: right,
      columns: (1.5em, 1.5em, 1.5em, 1.5em, 1.5em, 1.5em),
      rows:  (1.5em, 1.5em, 1.5em, 1.5em, 1.5em, 1.5em),
      [0], [1], [2], [3], [4], [5],
      [1], [👗], [ ], [ ], [ ], [🐶],
      [2], [👠], [👟], [ ], [🐻], [ ],
      [3], [💳], [💶], [ ], [ ], [ ],
      [4], [ ], [ ], [ ], [ ], [ ],
      [5], [ ], [ ], [ ], [ ], [🎸],
    ),
    grid.cell(inset: 0.5em, sym.arrow),
    table(
    align: right,
      columns: (1.5em, 1.5em, 1.5em),
      rows: (1.5em, 1.5em, 1.5em, 1.5em, 1.5em, 1.5em, 1.5em, 1.5em),
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
    grid.cell(
    align: bottom,
    inset: 1em,
    text(
    ["bear in a dress wearing heels" \
    \
    Dimensions \
    🐻 👗  👠 [[4,1], [0,0], [0,1]] \
    🐶 👗  👠 [[5,0], [0,0], [0,1]] \
    🐻 👗  👟 [[4,1], [0,0], [1,1]] \
    🐶 👗  👟 [[5,0], [0,0], [1,1]] \

    ]
    )
    )
  )

]

#slide[
  == The PostgreSQL table

```sql
CREATE EXTENSION vector;
```

```sql
CREATE TABLE pictures (
    filename text PRIMARY KEY,
    url text,
    embedding vector(512));
```
]

// - typing to make our code better
// - fastapi to give us a web app
// - jinja2 to allow us to create our HTML page template
// - python-dotenv to support taking environment variables from a .env file,
//   or from the environment
// Why "fastapi[standard]"? - it's what the fastapi installation guide says to use :)
#slide[
== Python packages we use (1)

Making the app work
- `typing`
- `fastapi[standard]`
- `jinja2`
- `python-dotenv`
]

// - If I was starting now, I'd use psycopg3, as it's nicer
// - There are various CLIP options we might choose - this is "the original"?
// - We need Torch to handle ...
#slide[
== Python packages we use (2)

Talking to PostgreSQL
- `psycopg[binary]`

The LLM bit
- `git+https://github.com/openai/CLIP.git`
- `torch`
]

// ==========================================================
// And now, too much code for one talk!!!
// ==========================================================

#slide[
  == And now, too much code for one talk

  ...get it in now, and remove it later as it's not needed...
]

#slide[
  == Populating the PostgreSQL tables

- Run a program
- `process_images.py`
]

#slide[
== The `dotenv` dance
```python
SERVICE_URI = os.getenv("DATABASE_URL")
if not SERVICE_URI:
    load_dotenv()
    SERVICE_URI = os.getenv("DATABASE_URL")
if not SERVICE_URI:
    import sys
    sys.exit('No value found for environment variable'
             ' DATABASE_URL (the PG database)')
```
]

#slide[
== Load the open CLIP model
```python
MODEL_NAME = 'ViT-B/32'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
```
]

// If we download it remotely, it will default to being cached in ~/.cache/clip
#slide[
== Load the open CLIP model (actually)
```python
MODEL_NAME = 'ViT-B/32'
LOCAL_MODEL = Path('./models/ViT-B-32.pt').absolute()

if LOCAL_MODEL.exists():
    model, preprocess = clip.load(
        MODEL_NAME, device=DEVICE,
        download_root=LOCAL_MODEL.parent)
else:
    model, preprocess = clip.load(
        MODEL_NAME, device=DEVICE)
```
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
== Compute features (1)
```python
def compute_clip_features(photos_batch):
    photos = [Image.open(photo_file)
        for photo_file in photos_batch]

    photos_preprocessed = torch.stack(
        [preprocess(photo) for photo in photos]).to(DEVICE)
```
]

#slide[
== Compute features (2)

```python
    with torch.no_grad():
        photos_features = model.encode_image(
            photos_preprocessed)
        photos_features /= photos_features.norm(
            dim=-1, keepdim=True)

    return photos_features.cpu().numpy()
```
]

// See https://www.psycopg.org/psycopg3/docs/basic/copy.html for more on
// the use of COPY.
#slide[
== Writing data rows to PostgreSQL
```python
with psycopg.connect(SERVICE_URI) as conn:
    with conn.cursor() as cur:
        with cur.copy(
            'COPY pictures (filename, url, embedding)'
            ' FROM STDIN') as copy:
            for row in data:
                copy.write_row(row)
```
]

#slide[
== Convert our embedding vector into an SQL string
```python
def vector_to_string(embedding):
    vector_str = ", ".join(
        str(x) for x in embedding.tolist()
    )
    vector_str = f'[{vector_str}]'
    return vector_str
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
== This is familiar
```python
LOCAL_MODEL = Path('./models/ViT-B-32.pt').absolute()
MODEL_NAME = 'ViT-B/32'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

...but we delegate _loading_ the model to a function
]

// The types of the values are found from the docstring for clip.load
//
// See also the source code at https://github.com/openai/CLIP/blob/main/clip/clip.py
//
// (we could just make them type Any, but it's interesting to know the actual types)
#slide[
== Let's define our Model, ready for lazy-loading
```python
@dataclass
class Model:
    model: Union[None, torch.nn.Module]
    preprocess: Union[None,
                      Callable[[PIL.Image], torch.Tensor]]
    error_string: str

clip_model = Model(
    None, None,
    "CLIP model not loaded yet - try again soon")
```
]

// And the function that does the loading
#slide[
```python
def load_clip_model():
    try:
        # The same code we saw earlier
    except Exception as exc:
        clip_model.error_string = 'Unable to load '
            'CLIP model - please restart the application'
        logger.exception(clip_model.error_string)
    else:
        logger.info('CLIP model imported')
```
]

// Define events at the start and end of the app lifespan
//
// This means we'll start the CLIP model loading before the app appears for the user,
// but won't block them from seeing the web page early.
//
// And we needed the Model structure so that we could know when the async
// model loading had finished
#slide[
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('Async load task starting')

    blocking_loader = asyncio.to_thread(load_clip_model)
    asyncio.create_task(blocking_loader)

    yield

    # We don't have an unload step
```
]

// Encode the text to compute the feature vector and normalize it
// This is _very similar_ to what we do for the images
#slide[
```python
def get_single_embedding(text):
    with torch.no_grad():
        text_input = clip.tokenize([text]).to(DEVICE)
        text_features = \
             clip_model.model.encode_text(text_input)
        text_features /= \
             text_features.norm(dim=-1, keepdim=True)

    # Return the feature vector
    return text_features.cpu().numpy()[0]
```
]

#slide[
```python
def search_for_matches(text):
    """Returns pairs of the form (image_name, image_url)"""
    logger.info(f'Searching for {text!r}')
    vector = get_single_embedding(text)

    embedding_string = vector_to_string(vector)
```
]

// I've left out error handling - if an exception occurs,
// it will return (None, None)
#slide[
```python
    # Perform search
    with psycopg.connect(SERVICE_URI) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT filename, url FROM pictures"
                " ORDER BY embedding <-> %s LIMIT 4;",
                (embedding_string,),
            )
            return cur.fetchall()
```
]

#slide[
== Make the app work
```python
app = FastAPI(lifespan=lifespan, redirect_slashes=False)
templates = Jinja2Templates(directory="templates")
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


// Remember to update the shortlink go.aiven.io/tibs-signup
#slide[
  == Fin

  #grid(
    rows: 2,
    columns: (auto, auto),
    align: left,
    row-gutter: 2em,
    column-gutter: 10.0pt,

    [
      Get a free trial of Aiven services at \
      https://go.aiven.io/pyconuk-clip

      // STRONG NOTE TO SELF: Remember to create the shortlink for pyconuk-clip !!!

      Also, we're hiring! See https://aiven.io/careers
    ],
    tiaoma.qrcode("https://go.aiven.io/5-kinds-of-db", options: (scale: 2.35)),

    [
      Slides created using
      #link("https://typst.app/")[typst] and
      #link("https://typst.app/universe/package/polylux/")[polylux],
      and available at
      https://github.com/Aiven-Labs/app-multimodal-search-CLIP-PostgreSQL/slides,
      licensed
      #box(
        baseline: 50%,
        image("images/cc-attribution-sharealike-88x31.png"),
      )
    ],

    tiaoma.qrcode("https://github.com/Aiven-Labs/app-multimodal-search-CLIP-PostgreSQL/slides", options: (scale: 2.0)),
  )

]
