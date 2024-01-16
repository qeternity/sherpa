import argparse
import time
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from aphrodite import AsyncEngineArgs, AsyncAphrodite

from .prompt import Prompt

torch.inference_mode()

engine = None


class GenerateRequest(BaseModel):
    prompt: str


app = FastAPI()
request_counter = Counter()


@app.post("/generate")
async def generate(req: GenerateRequest):
    t0 = time.time()

    request_id = str(next(request_counter))
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)
    context = await Prompt(tokenizer, generator, settings, req.prompt)()
    outputs = engine.generate(req.prompt, sampling_params, request_id)
    async for output in outputs:
        print(output)

    t1 = time.time()
    _sec = t1 - t0
    print(f"Generated {len(output.outputs[0].token_ids)} tokens in {_sec}")

    return JSONResponse({"done": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = AsyncEngineArgs.add_cli_args(parser)

    # parser.add_argument("--model", type=str, required=True, help="model path")
    parser.add_argument("--port", type=int, required=True, help="api port")
    # parser.add_argument("--batches", type=int, required=False, help="num batches")

    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncAphrodite.from_engine_args(engine_args)

    uvicorn.run(app, host="0.0.0.0", port=args.port)
