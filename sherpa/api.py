import asyncio
import json
import sys
import time
from typing import Any, Dict, List, Optional, Union

sys.path.append("/root/sherpa/exllamav2")
sys.path.append("/root/sherpa/guidance")

import elasticapm
import torch
import uvicorn
from elasticapm.contrib.starlette import ElasticAPM, make_apm_client
from exllamav2_hf import Exllamav2HF
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoConfig, AutoTokenizer

import guidance

# [init torch]:
torch.set_grad_enabled(False)
torch.cuda._lazy_init()
torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.set_printoptions(precision=10)

# Setup FastAPI:
app = FastAPI()
apm = make_apm_client(
    {
        "SERVER_URL": "https://apm.zuma.dev/",
        "SERVICE_NAME": "sauron-13b",
        "SECRET_TOKEN": "",
        "SPAN_COMPRESSION_EXACT_MATCH_MAX_DURATION": "0ms",
    }
)
app.add_middleware(ElasticAPM, client=apm)

semaphore = asyncio.Semaphore(1)


class GenerateRequest(BaseModel):
    message: str
    prompt: Optional[str] = None
    max_new_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.7
    top_k: Optional[int] = 20
    top_p: Optional[float] = 0.65
    min_p: Optional[float] = 0.06
    token_repetition_penalty_max: Optional[float] = 1.15
    token_repetition_penalty_sustain: Optional[int] = 256
    token_repetition_penalty_decay: Optional[int] = None
    stream: Optional[bool] = True
    # options:
    # break_on_newline: Optional[str] = None
    options: Optional[Dict[str, Any]] = {}


@app.post("/generate")
async def stream_data(req: GenerateRequest):
    with elasticapm.capture_span("acquire lock"):
        await semaphore.acquire()
    try:
        t0 = time.time()
        # output = guidance(req.prompt)(**req.options)
        with elasticapm.capture_span("guidance"):
            output = guidance(req.prompt, caching=False, accelerate=True)()
        t1 = time.time()
        _sec = t1 - t0
        print(f"Output generated in {_sec}")
        resp = output.variables()
        resp.pop("llm", None)
        resp.pop("@raw_prefix", None)
        resp.pop("logging", None)
        resp.pop("accelerate", None)
        return JSONResponse(
            [
                json.dumps(resp, indent=4),
            ]
        )
    finally:
        semaphore.release()


# -------


if __name__ == "__main__":
    config = AutoConfig.from_pretrained("/root/Nous-Hermes-Llama2-13b-GPTQ")
    tokenizer = AutoTokenizer.from_pretrained("/root/Nous-Hermes-Llama2-13b-GPTQ")
    model = Exllamav2HF.from_pretrained("/root/Nous-Hermes-Llama2-13b-GPTQ")
    model.config = config

    guidance.llm = guidance.llms.Transformers(
        model,
        tokenizer,
        caching=False,
        acceleration=False,
        device=0,
    )

    _PORT = 8001
    uvicorn.run(app, host="0.0.0.0", port=_PORT)
