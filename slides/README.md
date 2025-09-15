# Type text, find pictures: an app using CLIP, PostgreSQL® and pgvector

> **Note** Slides are still in draft.

Slides and notes for the talk "Type text, find pictures: an app using CLIP, PostgreSQL® and pgvector", a talk (to be) given at
[PyCon UK 2025](https://2025.pyconuk.org/)
on [Sunday, September 21st
2025](https://pretalx.com/pyconuk-2025/talk/VTHPZK/)
by [Tibs](https://aiven.io/Tibs).

This is a talk about the application that lives in this repository.

When the video recording is released, a link will be provided.


## Abstract

**What would once have been magical is now becoming common place.**

**In this talk, I'll show how to write a Python app that takes a text snippet 
(like "cat" or "man jumping") and finds images that match.**

The original LLMs (large language models) worked with text, making it possible to use a phrase to search for documents with a similar meaning. Later models addressed other media, for instance, allowing comparison of images. And now we have multi-modal models, like OpenAI's CLIP, which can work with images and text interchangably.

I'll give a quick introduction to vector search and CLIP, talk through setting up the necessary table in PostgreSQL®, walk through a script to calculate the embeddings of the chosen images, and store them in the database, and another script that takes a search text and uses pgvector to find matching images in the database.

I'll then show how you can use FastAPI and HTMX to quickly make a web app with a basic form. Merging the "find images" code into that then gives the final application.

All the code will also be available in a GitHub repository. _(that's this 
repository!)_

## Documents here

[slides.typ](slides.typ) is the Typst source code for the slides. The PDF for
the slides is [slides.pdf](slides.pdf). This version of the talk is aimed at 30
minutes or less.

[notes.md](notes.md) are the notes I made when writing the talk.

## Creating the slides

The slides are created using [typst](https://typst.app/) and the
[polylux](https://typst.app/universe/package/polylux/) package.

They use the Libertinus fonts - see https://github.com/alerque/libertinus.
 
To build the PDF I use the command line tool `typst`. See the [installation
instructions](https://github.com/typst/typst?tab=readme-ov-file#installation)
from the [typst GitHub repostory](https://github.com/typst/typst) - on my mac
I install it with `brew install typst` - and then
```shell
typst compile -f pdf slides.typ
```
 
The provided `Makefile`
can also produce the slides
(you'll need `make`, either GNU or BSD should work).
Try
```shell
make pdf
```
(this will run `typst` and then `open` the PDF) and `make help` for what else it can do.

Or if you prefer, you can use the equivalent `justfile` (you'll need
[just](https://just.systems/man/en/introduction.html))
```shell
just pdf
```

--------

![CC-Attribution-ShareAlike image](images/cc-attribution-sharealike-88x31.png)

This talk and its related files are released under a [Creative Commons
Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).
