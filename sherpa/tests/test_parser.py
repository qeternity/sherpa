import sys

sys.path.append("../..")

from sherpa.prompt import Prompt

prompt = """
The best part about the beach is {{gen "beach" stop_regex="[\\.\\n]" max_tokens="1000"}}

The best part about the mountains are {{ gen   "mountains" max_tokens="1000" }}
""".strip()

prompt_obj = Prompt(None, prompt)
prompt_obj._prepare()

print(prompt_obj.ops)
