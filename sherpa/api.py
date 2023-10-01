import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Union

import elasticapm
import torch
import uvicorn
from elasticapm.contrib.starlette import ElasticAPM, make_apm_client
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from generator import ExLlamaGenerator
from llama_cpp_guidance.llm import LlamaCpp
from model import ExLlama, ExLlamaCache, ExLlamaConfig
from pydantic import BaseModel
from tokenizer import ExLlamaTokenizer
from transformers import AutoTokenizer
from transformers.generation import SampleDecoderOnlyOutput
from transformers.generation.streamers import BaseStreamer

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
    from test_guidance import Model, Tokenizer

    # _config = ExLlamaConfig('/root/wizardLM-13B-1.0-GPTQ/config.json')
    # _config.model_path = '/root/wizardLM-13B-1.0-GPTQ/WizardLM-13B-1.0-GPTQ-4bit-128g.no-act-order.safetensors'
    # _tokenizer = ExLlamaTokenizer('/root/wizardLM-13B-1.0-GPTQ/tokenizer.model')
    # tokenizer = AutoTokenizer.from_pretrained('/root/wizardLM-13B-1.0-GPTQ/')
    # _config = ExLlamaConfig('/root/wizardLM-7B-GPTQ/config.json')
    # _config.model_path = '/root/wizardLM-7B-GPTQ/wizardLM-7B-GPTQ-4bit.compat.no-act-order.safetensors'
    # _tokenizer = ExLlamaTokenizer('/root/wizardLM-7B-GPTQ/tokenizer.model')
    # tokenizer = AutoTokenizer.from_pretrained('/root/wizardLM-7B-GPTQ/')
    # _config = ExLlamaConfig('/root/llama-7b-gptq-4bit-128g/config.json')
    # _config.model_path = '/root/llama-7b-gptq-4bit-128g/llama7b-gptq-4bit-128g.safetensors'
    # _tokenizer = ExLlamaTokenizer('/root/llama-7b-gptq-4bit-128g/tokenizer.model')
    # tokenizer = AutoTokenizer.from_pretrained('/root/llama-7b-gptq-4bit-128g/', use_fast=True)
    # _config = ExLlamaConfig('/root/Llama-2-13B-GPTQ/config.json')
    # _config.model_path = '/root/Llama-2-13B-GPTQ/gptq_model-4bit-128g.safetensors'
    # _tokenizer = ExLlamaTokenizer('/root/Llama-2-13B-GPTQ/tokenizer.model')
    # tokenizer = AutoTokenizer.from_pretrained('/root/Llama-2-13B-GPTQ/', use_fast=True)

    _config = ExLlamaConfig("/root/Nous-Hermes-Llama2-13b-GPTQ/config.json")
    _config.model_path = (
        "/root/Nous-Hermes-Llama2-13b-GPTQ/gptq_model-4bit-32g.safetensors"
    )
    _tokenizer = ExLlamaTokenizer("/root/Nous-Hermes-Llama2-13b-GPTQ/tokenizer.model")
    tokenizer = AutoTokenizer.from_pretrained(
        "/root/Nous-Hermes-Llama2-13b-GPTQ/", use_fast=True
    )

    # _config = ExLlamaConfig('/root/OpenOrca-Platypus2-13B-GPTQ/config.json')
    # _config.model_path = '/root/OpenOrca-Platypus2-13B-GPTQ/gptq_model-4bit-128g.safetensors'
    # _tokenizer = ExLlamaTokenizer('/root/OpenOrca-Platypus2-13B-GPTQ/tokenizer.model')
    # tokenizer = AutoTokenizer.from_pretrained('/root/OpenOrca-Platypus2-13B-GPTQ/', use_fast=True)
    # tokenizer = Tokenizer(_tokenizer, _config)
    # _config = ExLlamaConfig('/root/LLongMA-2-7B-GPTQ/config.json')
    # _config.model_path = '/root/LLongMA-2-7B-GPTQ/gptq_model-4bit-128g.safetensors'
    # _tokenizer = ExLlamaTokenizer('/root/LLongMA-2-7B-GPTQ/tokenizer.model')
    # tokenizer = AutoTokenizer.from_pretrained('/root/LLongMA-2-7B-GPTQ/')

    _model = ExLlama(_config)
    generator = ExLlamaGenerator(_model, _tokenizer, ExLlamaCache(_model))
    model = Model(generator)
    guidance.llm = guidance.llms.Transformers(
        model=model, tokenizer=tokenizer, caching=False
    )

    # from pathlib import Path
    # guidance.llm = LlamaCpp(
    #     model_path=Path("/root/Nous-Hermes-Llama2-GGUF/nous-hermes-llama2-13b.Q5_K_M.gguf"),
    #     n_gpu_layers=1000,
    #     n_threads=8
    # )

    # [start fastapi]:
    _PORT = 8001
    uvicorn.run(app, host="0.0.0.0", port=_PORT)
