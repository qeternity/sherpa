import argparse
import asyncio
from contextlib import asynccontextmanager
import json
import sys
import time
from typing import Any, Dict, List, Optional, Union

from prompt import Prompt
from prompt_async import Prompt as AsyncPrompt

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
from exllamav2.generator import ExLlamaV2StreamingGenerator, ExLlamaV2Sampler, ExLlamaV2BatchedGeneratorAsync, ExLlamaV2BatchedModelAsync
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

torch.inference_mode()

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="model path")
parser.add_argument("--port", type=int, required=True, help="api port")
parser.add_argument("--batches", type=int, required=False, help="num batches")

NUM_BATCHES = 1

tokenizer = None
model = None
cache = None
generator = None
settings = ExLlamaV2Sampler.Settings()
settings.temperature = 0
settings.top_k = 1
generator_queue = asyncio.Queue()


class GenerateRequest(BaseModel):
    prompt: str

@asynccontextmanager
async def lifespan(app):
    await generator_queue.put(generator)
    for i in range(NUM_BATCHES - 1):
        await generator_queue.put(ExLlamaV2BatchedGeneratorAsync(model, cache.clone(), tokenizer))

    yield

app = FastAPI(lifespan=lifespan)
# apm = make_apm_client(
#     {
#         "SERVER_URL": "https://apm.zuma.dev/",
#         "SERVICE_NAME": "sherpa",
#         "SECRET_TOKEN": "",
#         "SPAN_COMPRESSION_EXACT_MATCH_MAX_DURATION": "0ms",
#     }
# )
# app.add_middleware(ElasticAPM, client=apm)


@app.post("/generate")
async def stream_data(req: GenerateRequest):
    generator = await generator_queue.get()
    try:
        t0 = time.time()
        if NUM_BATCHES == 1:
            context = Prompt(tokenizer, generator, settings, req.prompt)()
        else:
            context = await AsyncPrompt(tokenizer, generator, settings, req.prompt)()
        t1 = time.time()
        _sec = t1 - t0
        print(f"Generated {context.token_count} tokens in {_sec}")
        return JSONResponse(context.vars)
    finally:
        await generator_queue.put(generator)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.batches:
        NUM_BATCHES = args.batches

    model_path = args.model
    config = ExLlamaV2Config()
    config.model_dir = model_path
    config.prepare()
    config.max_input_len = 8192
    config.max_attention_size = 8192**2
    tokenizer = ExLlamaV2Tokenizer(config)
    if NUM_BATCHES == 1:
        model = ExLlamaV2(config)
    else:
        model = ExLlamaV2BatchedModelAsync(config, NUM_BATCHES)
    cache = ExLlamaV2Cache(model, lazy=True)
    model.load_autosplit(cache)
    if NUM_BATCHES == 1:
        generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)
    else:
        generator = ExLlamaV2BatchedGeneratorAsync(model, cache, tokenizer)

    uvicorn.run(app, host="0.0.0.0", port=args.port)
