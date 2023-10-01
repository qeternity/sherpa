import sys

sys.path.append("/root/sherpa/exllamav2")
sys.path.append("/root/sherpa/guidance")
import guidance
from transformers import AutoConfig, AutoTokenizer
from exllamav2_hf import Exllamav2HF

config = AutoConfig.from_pretrained("/root/Nous-Hermes-Llama2-13b-GPTQ")
tokenizer = AutoTokenizer.from_pretrained("/root/Nous-Hermes-Llama2-13b-GPTQ")
model = Exllamav2HF.from_pretrained("/root/Nous-Hermes-Llama2-13b-GPTQ")
model.config = config

guidance.llm = guidance.llms.Transformers(
    model,
    tokenizer,
    device=0,
    caching=False,
    accelerate=False,
)

generated = guidance(
    """The best thing about the beach is {{~gen 'best' temperature=0.7 max_tokens=128}}"""
)()
print(generated)
