#!/usr/bin/env python3

"""An app to find (the first four) images matching a text string, and display them.
"""

import logging
import os

from typing import Annotated, Callable, Union

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
)

logger = logging.getLogger(__name__)


app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "search_hint": "Find images like...",
        },
    )


@app.post("/search_form", response_class=HTMLResponse)
async def search_form(request: Request, search_text: Annotated[str, Form()]):
    logging.info(f'Search form requests {search_text!r}')
    return templates.TemplateResponse(
        request=request,
        name="images.html",
        context={
            "images": [],
            "error_message": 'Search is not implemented in this version of the app',
        }
    )
