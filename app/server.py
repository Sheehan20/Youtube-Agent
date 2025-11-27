#!/usr/bin/env python
# -*- coding: utf-8 -*-
# YouTube Agent Server Module

import os
import sys
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from pydantic import BaseModel
from utils import create_workflow

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Initialize graph nodes workflow
chain = create_workflow(
    os.getenv('OPENAI_API_KEY'),
    os.getenv('model'),
    os.getenv('BASE_URL')
)


class Input(BaseModel):
    input: str


class Output(BaseModel):
    output: dict


app = FastAPI(
    title="YouTubeAgent Server",
    version="1.0",
    description="An API designed specifically for real-time retrieval of live data from YouTube."
)


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Add routes
add_routes(
    app,
    chain.with_types(input_type=Input, output_type=Output),
    path="/youtube_agent_chat",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
