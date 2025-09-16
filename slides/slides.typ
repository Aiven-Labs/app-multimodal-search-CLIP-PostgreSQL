// Get Polylux from the official package repository
#import "@preview/polylux:0.4.0": *

// Fletcher for diagrams
#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node
#import fletcher.shapes: chevron, pill

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
// (or "Helvetica Neue") and "Courier" for monospace text

#set text(font: "Libertinus Sans")

#show heading.where(
  level: 1,
): it => text(
  font: "Libertinus Serif",
  it.body,
)

#show heading.where(
  level: 2,
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
#let slide-footer = context [
  #set text(size: 15pt, fill: gray)
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
  if it.attribution != none [ -- #text(size: 20pt)[#it.attribution]]
}

// Give a very light grey background to code blocks, and make them all the same width
#show raw.where(block: true): it => block(
  // Previously I tried `fill: luma(240)`, but I actually prefer
  // the colour that rst2pdf uses for code block in slides
  fill: rgb("f0f8ff"),
  width: 35em,
  inset: 5pt,
  it,
)

// But be more subtle with inline code
#show raw.where(block: false): box.with(
  // I _think_ we can do without a fill colour for inline code terms
  inset: (x: 3pt, y: 0pt),
  outset: (y: 3pt),
)

// I don't want "Figure 1:" in the figure caption text
#show figure.caption: it => [
  #it.body
]

// Use #slide to create a slide and style it using your favourite Typst functions
#slide[
  #set align(horizon)

  // = Explaining the 5 types of database and how to choose between them

  #heading(
    level: 1,
    [Type text, find pictures: an app using CLIP, PostgreSQL® and pgvector],
  )

  #v(20pt)

  Tibs (they / he)

  #v(15pt)

  #grid(
    columns: (70%, 30%),
    align: (top, horizon),

    text(size: 22pt)[21#super[st] September 2025, PyCon UK 2025],

    align(right, grid(
      rows: (auto, auto),
      align: center,
      row-gutter: 10.0pt,
      tiaoma.qrcode("https://go.aiven.io/pyconuk-tibs", options: (scale: 3.0)),
      text(size: 20pt)[https://aiven.io/tibs],
    )),
  )
]

#slide[
  == What we're about
  #align(center)[
    #grid(
      columns: (auto, auto),
      gutter: 2em,
      box(
        image("images/app-cute-dog.png", width: 10em),
        clip: true,
        inset: (bottom: -6.2em),
      ),
      align(
        center + horizon,
      )[
        _This is where I\ show you a demo_
        /*_If we have the app,\ show the QR code\ and let people play.\
        Otherwise give a\ demo or play\ the sample video_*/
      ],
    )
  ]
]

#slide[
  == Software and slides

  #v(20pt)

  #grid(
    columns: (auto, auto),
    align: left,
    column-gutter: 1em,

    [
      All the code is available at \
      https://github.com/Aiven-Labs/app-multimodal-search-CLIP-PostgreSQL
    ],

    tiaoma.qrcode(
      "https://github.com/Aiven-Labs/app-multimodal-search-CLIP-PostgreSQL",
      options: (scale: 2.1),
    ),
  )

  #v(10pt)

  #grid(
    columns: (auto, auto),
    align: left,
    column-gutter: 1em,

    tiaoma.qrcode(
      "https://github.com/Aiven-Labs/app-multimodal-search-CLIP-PostgreSQL/slides/slides.pdf",
      options: (scale: 1.9),
    ),
    [
      The slides themselves are at \
      https://github.com/Aiven-Labs/app-multimodal-search-CLIP-PostgreSQL/slides/slides.pdf
    ],
  )
]

#slide[
  == Agenda

  - A brief introduction to LMM and OpenAI CLIP
  - Overview of how the app works
  - Setting up PostgreSQL
  - Calculating and storing image embeddings
  - Check: find an image!
  - Wrap that up into an app
]

#slide[
  #align(horizon + center)[
    #heading()[A brief introduction to\ LMM and OpenAI CLIP]
  ]
]

#slide[
  == I'm not going to explain vector embeddings

  ...but you can watch my PyCon UK 2023 talk for a simple introduction.

  #figure(
    caption: ["How I used PostgreSQL® to find pictures of me at a party"],
    align(center)[
      #grid(
        columns: (auto, auto),
        column-gutter: 3em,
        row-gutter: 20pt,

        image("images/tibs-pyconuk-2023.png"),

        grid(
          rows: 2,
          row-gutter: 10pt,
          align(horizon + center)[
            #tiaoma.qrcode(
              "https://www.youtube.com/watch?v=_FqKxKVJGWQ",
              options: (scale: 3.0),
            )
          ],
          align(center)[
            #text(size: 20pt)[
              #link("https://www.youtube.com/watch?v=_FqKxKVJGWQ")[
                `https://www.youtube.com/watch` \
                `?v=_FqKxKVJGWQ`
              ]
            ]
          ],
        ),
      )
    ],
  )


]

#slide[
  == Multimodal magic

  *Modalities* -- kinds of data

  Text, images, videos, audio, code, equations...

  - Old: convert data in any modality to text, and then use a (text) LLM

  - New: train on different kinds of data simultaneously,
    relating them using "fusion mechanisms"

  #sym.arrow.r.stroked "Large Multimodal Models"
]

#slide[
  == OpenAI CLIP

  https://github.com/openai/CLIP and https://openai.com/index/clip/

  #quote(block: true)[
    OpenAI's CLIP (Contrastive Language-Image Pre-training) is a neural network that is
    trained to understand images paired with natural language.
  ]

  - highly efficient
  - flexible and general
  - carefully limited in its ambitions (don't ask it to count things!)
]

#slide[
  == CLIP libraries

  We are using https://github.com/openai/CLIP

  ```shell
  pip install git+https://github.com/openai/CLIP.git
  ```

  OpenAI also suggest:

  - #link("https://github.com/mlfoundations/open_clip")[OpenCLIP]:
    includes larger and independently trained CLIP models up to ViT-G/14
  - #link("https://huggingface.co/docs/transformers/model_doc/clip")[Hugging Face
      implementation of CLIP]: for easier integration with the HF ecosystem

]

#slide[
  == Training CLIP: Contrastive pre-training

  #figure(
    image("images/CLIP-training-overview-a.svg", width: 15em),
    caption: text(size: 20pt)[source: https://github.com/openai/CLIP],
  )
]

#slide[
  == Make sure your vectors match

  An encoding assigns (some sort of) "meaning" to each number in an embedding

  - Don't try to match embeddings from one LLM against those from another

  Different models use different vector sizes

  - Don't try to match embeddings with different sizes

  _CLIP uses vectors with 512 elements_
]

#slide[
  #align(horizon + center)[
    #heading()[Overview of how the app works]
  ]
]

// Describe the overall flow of what we're doing
#slide[
  // And I want a smaller text size for the captions / diagram components
  #set text(size: 20pt)

  #diagram(
    // The default spacing between rows and columns is 3em, which is a bit
    // big for a slide, especially vertically with 3 rows
    spacing: (1.2em, 0.3em),

    // For debugging placement, it's useful to see the actual node
    //node-fill: teal.lighten(50%),

    // By default, nodes are rectangular or circular depending on their aspect
    // ratio. I want more control than that, so will make all nodes rectangular
    node-shape: rect,

    // Let's have a bit more gap between a node and its edge(s)
    node-outset: 5pt,

    node((0, 0), name: <photos>, figure(
      image("images/unsplash-dog-photo.png", width: 5em),
      caption: text(size: 20pt)[Photos from\ Unsplash],
    )),

    node(
      (2, 0),
      name: <clip-top>,
      figure(
        image("images/openai-clip.png", width: 3em),
        caption: text(size: 18pt)[CLIP model\ from OpenAI],
      ),
    ),

    node((4, 0), name: <vectors>, [Vectors in 512\ dimension space]),

    node((5, 2), name: <postgres>, grid(
      columns: (auto, auto),
      gutter: 0.5em,
      image("images/elephant.png", width: 3em), text(size: 20pt)[PostgreSQL],
    )),

    node((0, 4), name: <search>, image("images/search-phrase.png", width: 8em)),

    node((2, 4), name: <clip-bottom>, figure(
      image("images/openai-clip.png", width: 3em),
      caption: text(size: 18pt)[CLIP model\ from OpenAI],
    )),

    node((4, 4), name: <single-vector>, [Single vector]),

    edge(<photos>, "->", <clip-top>),
    edge(<clip-top>, "->", <vectors>),
    edge(<vectors>, "->", <postgres>),

    edge(<search>, "->", <clip-bottom>),
    edge(<clip-bottom>, "->", <single-vector>),
    edge(<single-vector>, "->", <postgres>),
  )
]

#slide[
  == What's in the repository

  #text(
    size: 24pt,
  )[https://github.com/Aiven-Labs/app-multimodal-search-CLIP-PostgreSQL]

  #grid(
    columns: (auto, auto),
    column-gutter: 90pt,

    grid(
      columns: (auto, auto),
      column-gutter: 40pt,
      [
        - `app.py`
        - `create_table.py`
        - `find_images.py`
        - `process_images.py`
        - `templates/`
      ],

      [
        - `photos/`
        - `slides/`
        - `Dockerfile`
        - `LICENSE`
        - `README.md`
        - `requirements.txt`
      ],
    ),

    tiaoma.qrcode(
      "https://github.com/Aiven-Labs/app-multimodal-search-CLIP-PostgreSQL",
      options: (scale: 2.5),
    ),
  )
]

#slide[
  == The `photos/` directory: 1000 pictures

  #align(center)[
    #image("images/photo-grid.png", width: 18em)
  ]
]

#slide[
  == Images from Unsplash

  http://unsplash.com

  #v(20pt)

  - Provided to make the repository "self-sufficient"
  - Images reduced in size to fit in GitHub
  - The example app refers to them in GitHub (using `<img>`)

  #v(20pt)

  For production, don't use GitHub as an image cache!
]

// - `create_table.py`   -- create the table we need in PG
// - `process_images.py` -- calculate the image embeddings and put them into PG
// - `find_images.py`    -- for testing it all works
// - `app.py`            -- the thing itself
#slide[
  == The app and related programs

  We separate _preparing the database_ from _using the app_

  - `create_table.py`
  - `process_images.py`
  - `find_images.py`
  - `app.py`
]

// - typing to make our code better
// - fastapi to give us a web app
//   (this uses starlette to provide the HTMLResponse class, and do the
//   HTML templating)
// - jinja2 to allow us to create our HTML page template
// - python-dotenv to support taking environment variables from a .env file,
//   or from the environment
//
// - Note that `psycopg` is now `psycopg3`, which is a lot nicer than the
//   older `psycopg2`
// - There are various CLIP options we might choose - this is "the original"
//   - `git+https://github.com/openai/CLIP.git`
// - We need Torch to handle ...
#slide[
  == The program requirements

  - `fastapi` for our web app

    - `jinja2` to handle our HTML templating

  - `psycopg` to talk to PostgreSQL#super[®]

  - `clip` from OpenAI to talk to the CLIP model

  - `torch` to handle some ML related computations // try to explain this better

  - and Pillow (`PIL`) to handle images
]

#slide[
  #align(horizon + center)[
    #heading()[Setting up PostgreSQL]
  ]

  #align(center + horizon)[`create_table.py`]
]

#set page(header: context [
  #set text(size: 15pt, fill: gray)
  #align(left)[`create_table.py`]
])

#slide[
  == Running things: Create the PostgreSQL table

  ```shell
  $ ./create_table.py
  Enabling pgvector
  Creating table
  Done
  ```
]

#slide[
  == Enable pgvector

  Enable the #link("https://github.com/pgvector/pgvector")[`pgvector`] extension:

  ```sql
  CREATE EXTENSION vector;
  ```

  This only works if the `pgvector` extension is installed.

  It may already be available, as in Aiven for PostgreSQL#super[®]

]

#slide[
  == Create our database table

  ```sql
  CREATE TABLE pictures (
      filename text PRIMARY KEY,
      url text,
      embedding vector(512));
  ```

  #v(2em)

  _Remember, CLIP uses vectors with 512 elements_
]

#set page(header: context [])

#slide[
  #align(horizon + center)[
    #heading()[Calculating and storing image embeddings]
  ]

  #align(center + horizon)[`process_images.py`]
]

#set page(header: context [
  #set text(size: 15pt, fill: gray)
  #align(left)[`process_images.py`]
])

#slide[
  == Running things: Calculate and store the image embeddings

  ```shell
  $ ./process_images.py
  Importing CLIP model ViT-B/32
  Using cpu
  Batch 0
  Batch 100
  ...
  Batch 900
  All embeddings indexed successfully.
  ```
]

// For both text and image (so process_images.py and app.py)
//
// This is assuming that we let `clip.load` automatically download
// the model file when it is first run.
// It will default to being cached in ~/.cache/clip.
//
// This is OK for process_images.py, but not something we want to
// do for the actual app, otherwise the first user query would have
// a LONG wait while the model is downloaded - and sometimes that
// download fails, which would mean we'd have to give the user an
// error message :(
// So for the app we do lazy loading - see later if we have time.
#slide[
  == Load the open CLIP model
  ```python
  MODEL_NAME = 'ViT-B/32'
  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

  model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
  ```

  #v(30pt)

  - `model` -- a `torch.nn.Module`

  - `preprocess` -- a `Callable[ [PIL.Image], torch.Tensor ]`
]

#slide[
  == Process a batch of photos (1)
  ```python
  def compute_clip_features(photos_batch):
      photos = [Image.open(photo_file)
                  for photo_file in photos_batch]

      photos_preprocessed = torch.stack(
          [preprocess(photo)
              for photo in photos]).to(DEVICE)
  ```
]

#slide[
  == Process a batch of photos (2)

  ```python
      with torch.no_grad():
          photos_features = model.encode_image(
              photos_preprocessed
          )
          photos_features /= photos_features.norm(
              dim=-1,
              keepdim=True
          )

      return photos_features.cpu().numpy()
  ```
]

#slide[
  == Convert an embedding vector into an SQL string
  ```python
  def vector_to_string(embedding):
      vector_str = ", ".join(
          str(x) for x in embedding.tolist()
      )
      vector_str = f'[{vector_str}]'
      return vector_str
  ```

  Converts a NumPy `ndarray` to a string:

  `[1.88626274e-02, ..., -2.86908299e-02], dtype=float32)` \
  #sym.arrow.r.stroked `"[0.01886262744665146, ..., -0.028690829873085022]"`
]

#slide[
  == Write data rows to PostgreSQL
  ```python
  with psycopg.connect(SERVICE_URI) as conn:
      with conn.cursor() as cursor:
          with cursor.copy('COPY pictures'
                           ' (filename, url, embedding)'
                           ' FROM STDIN') as copy:
              for row in data:
                  copy.write_row(row)
  ```
]

#slide[

  == That SQL query

  Copy data to the table (`pictures`) from the client (`STDIN`)
  ```sql
  COPY pictures (filename, url, embedding) FROM STDIN;
  ```
  This is one of the most efficient ways to load data into the database.

  See https://www.postgresql.org/docs/current/sql-copy.html for more
]

#slide[

  == SQL COPY context manager

  `psycopg` provides a context manager to help with this:
  ```python
  with cursor.copy('COPY pictures (filename, url,'
                   ' embedding) FROM STDIN') as copy:
      for row in data:
          copy.write_row(row)
  ```

  See https://www.psycopg.org/psycopg3/docs/basic/copy.html for more
]

#set page(header: context [])

#slide[
  #align(horizon + center)[
    #heading()[Check things are working\ Find some images]

    #align(center + horizon)[`find_images.py`]
  ]
]

#set page(header: context [
  #set text(size: 15pt, fill: gray)
  #align(left)[`find_images.py`]
])

#slide[
  == Running things: `find_images.py`

  ```shell
  $ ./find_images.py
  2025-09-15 16:15:36,875 __main__ INFO: Importing CLIP model
  2025-09-15 16:15:36,875 __main__ INFO: Using cpu
  2025-09-15 16:15:39,259 __main__ INFO: Searching for 'man jumping'
  1: tmKn9xNHY6I.jpg
  2: rmc8Wjr3-c8.jpg
  3: A9vhHNsJG4Y.jpg
  4: IW7yli35qDE.jpg
  ```
]

// Encode the text to compute the feature vector and normalize it
// This is _very similar_ to what we do for the images
#slide[
  == Get the embedding for the user's text string
  ```python
  def get_single_embedding(text):
      with torch.no_grad():
          text_input = clip.tokenize([text]).to(DEVICE)
          text_features = \
               clip_model.model.encode_text(text_input)
          text_features /= \
               text_features.norm(dim=-1, keepdim=True)

      return text_features.cpu().numpy()[0]
  ```
]

#slide[
  == Find matches
  ```python
  def search_for_matches(text):
      vector = get_single_embedding(text)
      embedding_string = vector_to_string(vector)
      with psycopg.connect(SERVICE_URI) as conn:
          with conn.cursor() as cur:
              cur.execute(
                  "SELECT filename, url FROM pictures"
                  " ORDER BY embedding <-> %s LIMIT 4;",
                  (embedding_string,))
              return cur.fetchall()
  ```
]

#slide[
  == That SQL query

  ```sql
  SELECT filename, url FROM pictures
      ORDER BY embedding <-> [0.3816255331, ..., 0.200309]
      LIMIT 4;
  ```

  - `<->` for the nearest results by L2 (euclidean) distance.
  - `<=>` for cosine similarity - it compares the angle/direction
  - `<#>` for the inner product - do the vectors point the same way
  - `<+>` for the L1 ("Manhattan" or "taxi cab") distance

]

#set page(header: context [])

#slide[
  #align(horizon + center)[
    #heading()[Make an application]
  ]

  #align(center + horizon)[`app.py`]
]

#set page(header: context [
  #set text(size: 15pt, fill: gray)
  #align(left)[`app.py` and `templates/`]
])

#slide[
  == Running things: Run the app

  In development mode:
  ```shell
  $ fastapi dev app.py
  ```

  ...or we might use Docker or another container mechanism

]

#slide[
  == The initial prompt

  #v(2em)

  #align(center)[
    #image("images/app-start-page.png")
  ]
]

#slide[
  == GET the prompt
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
  == `templates/index.html`

  (the interesting bits!)
  ```html
  <p></p>

  <form hx-post="/search_form" hx-target="#response">
  	<input type="text" name="search_text"
  	       placeholder="{{ search_hint }}">
    <button>Search</button>
  </form>

  <div id="response"><p>Nothing to see here yet</p></div>
  ```
]

// We'll cheat by re-using the "CLIP not loaded" image and cropping off the bottom...
#slide[
  == Making a query

  #v(2em)

  #align(center)[
    #box(
      image("images/app-CLIP-not-loaded.png"),
      clip: true,
      inset: (bottom: -2em),
    )
  ]
]

// Type hinting and logging left out for simplicity
#slide[
  == POST the results
  ```python
  @app.post("/search_form", response_class=HTMLResponse)
  async def search_form(request, search_text):
      if not clip_model.model:
          return # ==> Error message

      results = search_for_matches(search_text)
      return     # ==> Success
  ```
  "`Error message`" and "`Success`" code on the next slides
]

// No CLIP model yet, so no results, just an error
#slide[
  == POST the results: Error message
  ```python
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

// Success - we have results, and our error message is empty
#slide[
  == POST the results: Success
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

// Show the details of "success" on the next slide
#slide[
  == `templates/images.html`

  ```html
  <div id="images">
  	{% if error_message %}
  	    <p> {{ error_message }} </p>
  	{% else %}
  	    <!-- Show the images -->
  	{% endif %}
  </div>
  ```
  "`Show the images`" code on the next slide
]

#slide[
  == `templates/images.html` -- show the images
  ```html
  	{% for item in images %}
  	    <p>Image {{ loop.index }}: {{ item[0] }}</p>
  	    <p><img src="{{ item[1] }}"
  	            alt="Image {{ item[0] }}"></p>
  	{% else %}
  	    <p>No results found</p>
  	{% endfor %}
  ```
]

#slide[
  #align(center)[
    #box(
      image("images/app-man-jumping.png"),
      clip: true,
      inset: (bottom: -2em),
    )
  ]
]

// ==================================================================

#set page(header: context [
  #set text(size: 15pt, fill: gray)
  #align(left)[`app.py`]
])

#slide[
  #align(horizon + center)[
    #heading()[Lazy loading the CLIP model]
  ]

  #align(center + horizon)[`app.py`]
]

#slide[
  == CLIP model not loaded yet

  #v(2em)

  #align(center)[
    #image("images/app-CLIP-not-loaded.png")
  ]
]

// For both text and image (so process_images.py and app.py)
#slide[
  == Load the open CLIP model - we saw this earlier
  ```python
  MODEL_NAME = 'ViT-B/32' ww
  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

  model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
  ```
  ...but this downloads the CLIP model at runtime.

  #align(right)[(although it does then cache it)]

  Can we do better?
]

#slide[
  == Let's define where we'll keep the model file
  ```python
  LOCAL_MODEL = Path('./models/ViT-B-32.pt').absolute()
  ```
]

// The types of the values are found from the docstring for clip.load
//
// See also the source code at https://github.com/openai/CLIP/blob/main/clip/clip.py
//
// (we could just make them type Any, but it's interesting to know the actual types)
#slide[
  == Let's define a Model to hold our model data
  ```python
  @dataclass
  class Model:
      model: Union[None, torch.nn.Module]
      preprocess: Union[None,
                        Callable[[PIL.Image],
                                  torch.Tensor]]
      error_string: str
  ```
]

#slide[
  == And create a (global) instance

  ```python
  clip_model = Model(
      None,
      None,
      "CLIP model not loaded yet - try again soon")
  ```
]

// And the function that does the loading.
//
// We *could* (and maybe should) unset the `clip_model.error_string` if
// we succeed in importing the model - but instead we just rely on the
// code checking to see if `clip_model.model` is set, and assuming that
// in that case it won't look at the `error_string`.
#slide[
  == Load the model with a function
  ```python
  def load_clip_model():
      try:
          # Load the model
      except Exception as exc:
          clip_model.error_string = 'Unable to load CLIP'
              ' model - please restart the application'
          logger.exception(clip_model.error_string)
      else:
          logger.info('CLIP model imported')
  ```
  "`Load the model`" code on the next slide
]

#slide[
  == Load the model - the actual code
  ```python
      if LOCAL_MODEL.exists():
          clip_model.model, clip_model.preprocess = \
              clip.load(
                  MODEL_NAME, device=DEVICE,
                  download_root=LOCAL_MODEL.parent)
      else:
          clip_model.model, clip_model.preprocess = \
              clip.load(
                  MODEL_NAME, device=DEVICE)
  ```
]

// Define events at the start and end of the app lifespan
//
// This means we'll start the CLIP model loading before the app appears for the user,
// but won't block them from seeing the web page early.
//
// And we needed the Model structure so that we could know when the async
// model loading had finished
//
// Note we're only adding a step *before* the main app runs
// - we don't need anything at the end, so there's nothing after the yield
#slide[
  == And handle that as part of the `fastapi` lifespan
  ```python
  @asynccontextmanager
  async def lifespan(app: FastAPI):

      blocking_loader = asyncio.to_thread(load_clip_model)
      asyncio.create_task(blocking_loader)

      yield

  # Join things up
  app = FastAPI(lifespan=lifespan, redirect_slashes=False)
  ```
]

#slide[
  == Better model loading

  1. As fastapi starts up, it starts the model loading

    (and then carries on doing its thing)

  2. If the model file is already there, that completes quickly

  3. Otherwise, the separate thread loads the model

  4. Queries give a warning if the model is not loaded yet

]

#set page(header: context [])

#slide[
  == Next steps?

  - Try running it yourself!

  - Use a different CLIP library, or different CLIP models

  - Use different images

  - Use a different image storage strategy, and cache images

  - Cache query results (perhaps using Valkey) - does it make a difference?

  - Look at OpenSearch instead of PostgreSQL

]

#slide[
  == Aiven

  I work for Aiven (https://aiven.io/)

  //  #align(center)[_Your AI-ready Open Source Data Platform_]

  #align(center)[
    #grid(
      align: left,
      columns: 90%,
      [
        _Your AI-ready Open Source Data Platform_

        Aiven is an AI-ready open source data platform that combines open-choice services
        to rapidly stream, store and serve data across major cloud providers — simply and securely.
      ]
    )
  ]

  #v(1em)
  If you want to run this app yourself, you can use our free version of
  PostgreSQL, with `pgvector` already installed.
]


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
      https://go.aiven.io/pyconuk-clip-trial

      // STRONG NOTE TO SELF: Remember to create the shortlink for pyconuk-clip !!!

      Also, we're hiring! See https://aiven.io/careers
    ],
    tiaoma.qrcode("https://go.aiven.io/pyconuk-clip-trial", options: (
      scale: 2.35,
    )),

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

    tiaoma.qrcode(
      "https://github.com/Aiven-Labs/app-multimodal-search-CLIP-PostgreSQL/slides",
      options: (scale: 2.0),
    ),
  )
]
