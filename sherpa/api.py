import argparse
import asyncio
import json
import sys
import time
from typing import Any, Dict, List, Optional, Union

from prompt import Prompt

sys.path.append("/root/sherpa/exllamav2")

import elasticapm
import torch
import uvicorn
from elasticapm.contrib.starlette import ElasticAPM, make_apm_client
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Cache,
    ExLlamaV2Config,
    ExLlamaV2Tokenizer,
)
from exllamav2.generator import ExLlamaV2StreamingGenerator, ExLlamaV2Sampler
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

torch.inference_mode()

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="model path")
parser.add_argument("--port", type=int, required=True, help="api port")

# Setup FastAPI:
app = FastAPI()
# apm = make_apm_client(
#     {
#         "SERVER_URL": "https://apm.zuma.dev/",
#         "SERVICE_NAME": "sherpa",
#         "SECRET_TOKEN": "",
#         "SPAN_COMPRESSION_EXACT_MATCH_MAX_DURATION": "0ms",
#     }
# )
# app.add_middleware(ElasticAPM, client=apm)

semaphore = asyncio.Semaphore(1)
tokenizer = None
model = None
cache = None
generator = None
settings = ExLlamaV2Sampler.Settings()
settings.temperature = 0
settings.top_k = 1


class GenerateRequest(BaseModel):
    prompt: str


@app.post("/generate")
async def stream_data(req: GenerateRequest):
    with elasticapm.capture_span("acquire lock"):
        await semaphore.acquire()
    try:
        t0 = time.time()
        context = Prompt(tokenizer, generator, settings, req.prompt)()
        t1 = time.time()
        _sec = t1 - t0
        print(f"Generated {context.token_count} tokens in {_sec}")
        return JSONResponse(context.vars)
    finally:
        semaphore.release()


# -------


if __name__ == "__main__":
    args = parser.parse_args()
    model_path = args.model
    config = ExLlamaV2Config()
    config.model_dir = model_path
    config.prepare()
    tokenizer = ExLlamaV2Tokenizer(config)
    model = ExLlamaV2(config)
    cache = ExLlamaV2Cache(model, lazy=True)
    model.load_autosplit(cache)
    generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)

    uvicorn.run(app, host="0.0.0.0", port=args.port)
