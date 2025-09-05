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
// (or "Helvetica Neue")

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
  fill: luma(240),
  width: 35em,
  inset: 5pt,
  it,
)

// But be more subtle with inline code
#show raw.where(block: false): box.with(
  fill: luma(240),
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

  #grid(
    columns: 2,

    text(size: 20pt)[

      21#super[st] September 2025, PyCon UK 2025

      Slides available at https://github.com/Aiven-Labs/app-multimodal-search-CLIP-PostgreSQL/slides
    ],

    align(right, grid(
      rows: (auto, auto),
      align: center,
      row-gutter: 10.0pt,
      tiaoma.qrcode("https://aiven.io/tibs", options: (scale: 3.0)),
      text(size: 20pt)[https://aiven.io/tibs],
    )),
  )
]

// ================================================================

#slide[

  #set page(fill: yellow)

  == What I promised to talk about: Mostly done

  - I'll give a quick introduction to vector search and CLIP,
  - talk through setting up the necessary table in PostgreSQL®,
  - walk through a script to calculate the embeddings of the chosen images,
  - and store them in the database,
  - and another script that takes a search text and uses pgvector to find matching
    images in the database.

  (hmm - not so much walking through as it stands at the moment)
]

#slide[

  #set page(fill: yellow)

  == What I promised to talk about: Still to do

  - I'll then show how you can use FastAPI and HTMX to quickly make a web app
    with a basic form.
  - Merging the "find images" code into that then gives the final application.


  So basically I need to show more of:

  1. Introduce useful things
  2. Put them together to give programs
]

// ================================================================

#slide[
  == What we're about

  #align(center)[_A picture of the app finding "cute dog"_]

  #align(center)[_If we have the app, show the QR code and let people play_]
]

/*
#slide[
== Possible agenda

- If we have the app, show the QR code and let people play
- Brief introduction to LMM / vector embeddings
  - Brief introduction to multimodal stuff (magic!)
  - Digression on OpenAI CLIP and choosing which implementation to use

- Describe app structure / components
  - We separate _preparing the database_ from _using the app_
  - Explain some core code components

_Is there any other code that is notable enough to talk about, given time?_
  - Lazy loading the model at run time of the app, and/or downloading the model during `Dockerfile` setup
  - "Storing" the images on GitHub ("don't do that") so they're easy to display in HTML
  - _Show_ the GET / POST methods - or at least their outline/docs

- If we have the app, then _maybe_ show the Dockerfile and how to run the app
]
*/

#slide[
  == Possible agenda

  - Brief introduction to LMM / vector embeddings
  - Describe app structure / components

  _Is there any other code that is notable enough to talk about, given time?_

  - If we have the app, then _maybe_ show the Dockerfile and how to run the app
]

#slide[
  == Brief introduction to LMM / vector embeddings
]

// I like this slide - can I keep this slide? Even though I've used it before?
#slide[
  == ... not an explanation of ML

  #figure(
    image("images/markus-winkler-f57lx37DCM4-unsplash.jpg", width: 55%),
    caption: text(size: 15pt)[
      Photo by #link("https://unsplash.com/@markuswinkler")[Markus Winkler]
      on #link("https://unsplash.com/photos/f57lx37DCM4")[Unsplash]
    ],
  )
]

#slide[

  == Vectors and embeddings

  ML people talk about vectors and embeddings and vector embeddings.

  - A vector is an array of numbers representing a direction and a size.

  - "Embedding" means representing something in a computer.

  So a *vector embedding* is

  - an array of numbers representing a direction and size
  - stored in a computer.
]

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

  and we can do maths between vectors - for instance

  #quote(block: true)[
    Is #highlight(fill: luma(230))[the vector between colour 1 and colour 2]
    _similar to_
    #highlight(fill: luma(230))[the vector between colour 3 and colour 4]?
  ]
]

#slide[
  == How do we calculate the vectors?

  Back in the early days of Natural Language Processing, this was done by hand.

  Nowadays we use Machine Learning and especially Large Language Models.
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

/*
// Jay's extension of Olena's grid, with "Dimensions"
// Is this a slide I can use / explain?

// Also, I really should create this slide using the data from the previous slide,
// rather than copying the whole thing!
#slide[

  #grid(
    columns: (auto, auto, auto, auto),
    align: horizon,
    table(
    align: right,
      columns: (1.5em, 1.5em, 1.5em, 1.5em, 1.5em, 1.5em, 1.5em),
      rows:  (1.5em, 1.5em, 1.5em, 1.5em, 1.5em, 1.5em, 1.5em),
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
*/

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
    trained to understand images paired with natural language.]

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

// If I understand this correctly, this is actually doing the opposite of
// what we're doing - it's taking an image and returning a corresponding text
// or description
#slide[
  == Training CLIP: Create dataset classifier, predict...

  #figure(
    image("images/CLIP-training-overview-b.svg", width: 14em),
    caption: text(size: 20pt)[source: https://github.com/openai/CLIP],
  )
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
  == Describe app structure / components
  - We separate _preparing the database_ from _using the app_

  - `create_table.py`
  - `process_images.py`
  - `find_images.py` -- for testing
  - `app.py` -- the thing itself
]

// - typing to make our code better
// - fastapi to give us a web app
// - jinja2 to allow us to create our HTML page template
// - python-dotenv to support taking environment variables from a .env file,
//   or from the environment
//
// - Note that `psycopg` is now `psycopg3`, which is a lot nicer than the
//   older `psycopg2`
// - There are various CLIP options we might choose - this is "the original"?
//   - `git+https://github.com/openai/CLIP.git`
// - We need Torch to handle ...
#slide[
  == The program requirements

  - `fastapi` for our web app

  - `jinja2` to handle our HTML templating

  - `psycopg` to talk to PostgreSQL#super[®]

  - OpenAI `clip` to talk to the CLIP model

  - `torch` to handle some ML related computations // try to explain this better
]

#slide[
  == Let's talk about code
]

#slide[
  == Enable pgvector

  Enable the #link("https://github.com/pgvector/pgvector")[pgvector]:

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

  - `filename` is the name of the image file
  - `url` is the URL that we can use to show it in an `<img>` tag
  - `embedding` is the vector for this image

  CLIP uses vectors with 512 elements
]

// For both text and image (so process_images.py and app.py)
#slide[
  == Load the open CLIP model
  ```python
  MODEL_NAME = 'ViT-B/32'
  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

  model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
  ```
]

// But actually we don't really *want* to let the CLIP package
// lazily download the model, as that makes the first query very slow
// (and may fail). So we allow for the model to have been downloaded
// beforehand.
//
// (Note: If we let `clip` download it, it will default to being cached in ~/.cache/clip)
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

#slide[
  == Process a batch of photos (1)
  ```python
  photos = [Image.open(photo_file)
              for photo_file in photos_batch]

  photos_preprocessed = torch.stack(
      [preprocess(photo) for photo in photos]).to(DEVICE)
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

// See https://www.postgresql.org/docs/current/sql-copy.html for COPY .. FROM STDIN
// See https://www.psycopg.org/psycopg3/docs/basic/copy.html for how to use it in Python
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

// ==================================================================

#slide[

  #set page(fill: yellow)

  == If there's time, talk about lazy loading the model at run time of the app, and/or downloading the model during `Dockerfile` setup

  MAYBE MAYBE MAYBE
]

// For both text and image (so process_images.py and app.py)
#slide[
  == Load the open CLIP model - we saw this earlier
  ```python
  MODEL_NAME = 'ViT-B/32' ww
  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

  model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
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

// And the function that does the loading.
// I've left out error handling - naughty me - but it basically raises an
// logs the exception and sets `clip_model.error_string` to a message saying
// "Unable to load CLIP model - please restart the application"
#slide[
  ```python
  def load_clip_model():
      if LOCAL_MODEL.exists():
          logger.info(f'Importing CLIP model {MODEL_NAME} from {LOCAL_MODEL.parent}')
          logger.info(f'Using {DEVICE}')
          clip_model.model, clip_model.preprocess = clip.load(MODEL_NAME, device=DEVICE, download_root=LOCAL_MODEL.parent)
      else:
          logger.info(f'Importing CLIP model {MODEL_NAME}')
          logger.info(f'Using {DEVICE}')
          clip_model.model, clip_model.preprocess = clip.load(MODEL_NAME, device=DEVICE)
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

#slide[
  == Join things up...
  ```python
  app = FastAPI(lifespan=lifespan, redirect_slashes=False)
  ```
]


// ==================================================================

#slide[

  #set page(fill: yellow)

  - If there's time, maybe talk about "storing" the images on GitHub ("don't do that") so they're easy to display in HTML

  The images in the photos directory came from Unsplash and have been reduced in size to make them fit within GitHub filesize limits for a repository.

  Note When the app is running, it retrieves the images for the HTML page directly from this GitHub repository. This is not good practice for a production app, as GitHub is not intended to act as an image repository for web apps.


]

// ==================================================================

#slide[

  #set page(fill: yellow)

  - If there's time, just _show_ the GET / POST methods - or at least their outline/docs
]

// ==================================================================

#slide[

  #set page(fill: yellow)
  
  == Dockerfile
  ```
  FROM python:3.11-slim

  WORKDIR /app

  COPY ./requirements.txt /app

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
  ```

]

// ==================================================================



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

    tiaoma.qrcode(
      "https://github.com/Aiven-Labs/app-multimodal-search-CLIP-PostgreSQL/slides",
      options: (scale: 2.0),
    ),
  )

]
