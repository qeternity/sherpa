import argparse
import asyncio
from contextlib import asynccontextmanager
import json
import sys
import time
from typing import Any, Dict, List, Optional, Union

from prompt_async import Prompt as AsyncPrompt

sys.path.append("/root/sherpa/exllamav2")

import elasticapm
import torch
import uvicorn
from elasticapm.contrib.starlette import ElasticAPM, make_apm_client
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

torch.inference_mode()

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, required=True, help="api port")


class GenerateRequest(BaseModel):
    prompt: str

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


@app.get("/healthz")
async def stream_data():
    return JSONResponse({"status": "gucci"})


@app.post("/generate")
async def stream_data(req: GenerateRequest):
    t0 = time.time()
    context = await AsyncPrompt(req.prompt)()
    t1 = time.time()
    _sec = t1 - t0
    print(f"Generated {context.token_count} tokens in {_sec}")
    return JSONResponse(context.vars)
    


if __name__ == "__main__":
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
