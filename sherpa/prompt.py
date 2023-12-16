import sys
import re

sys.path.append("../../exllamav2/")

from typing import Any, Dict, Optional

# from exllamav2.exllamav2.generator import ExLlamaV2StreamingGenerator

# returns groups for cmd, name and args
op_matcher = re.compile(
    r"""(?P<pre>[^\{]*){{\s*(?P<cmd>[a-z]+)\s*\"(?P<name>\w+)\"(?P<args>[^\}]*)}}(?P<post>[^\{]*)"""
)
args_matcher = re.compile(r"""\s*(?P<key>\w+)=\"(?P<value>[^"]+)\"""")


class Op:
    pass


class NamedOp(Op):
    def __init__(self, name: str) -> None:
        self.name = name


class Echo(Op):
    def __init__(self, value: str) -> None:
        super().__init__()
        self.value = value

    def __repr__(self) -> str:
        return f"""Echo(value="{self.value.encode("unicode_escape").decode()}")"""

    def run(self, generator: Any) -> str:
        pass


class Generate(NamedOp):
    def __init__(
        self,
        name: str,
        max_tokens: Optional[str] = None,
        stop_regex: Optional[str] = None,
    ) -> None:
        super().__init__(name)
        self.max_tokens = int(max_tokens.strip()) if max_tokens else None
        self.stop_regex = stop_regex

    def __repr__(self) -> str:
        return f"Generate(name={self.name}, max_tokens={self.max_tokens}, stop_regex={self.stop_regex})"

    def run(self, tokenizer: Any, generator: Any, settings: Any, prompt: str) -> None:
        input_ids = tokenizer.encode(prompt)
        generator.begin_stream(input_ids, settings, token_healing=True)

        cnt = 0
        draft = ""
        while True:
            chunk, eos, _ = generator.stream()
            cnt += 1
            draft += chunk
            if eos or cnt == self.max_tokens:
                break
        return draft


class Prompt:
    # def __init__(self, generator: ExLlamaV2StreamingGenerator, prompt: str) -> None:
    def __init__(self, tokenizer, generator, settings, prompt: str) -> None:
        self.draft = ""
        self.tokenizer = tokenizer
        self.generator = generator
        self.settings = settings
        self.prompt = prompt

        self.output: Dict[str, str] = dict()
        self.ops = list()

    def __call__(self) -> Dict[str, str]:
        self._prepare()
        self._run()
        return self.output

    def _prepare(self) -> None:
        for op_match in op_matcher.finditer(self.prompt):
            pre = op_match.group("pre")
            cmd = op_match.group("cmd")
            name = op_match.group("name")
            args = op_match.group("args")
            post = op_match.group("post")

            if pre:
                self.ops.append(Echo(pre))

            if cmd == "gen":
                klass = Generate
            else:
                raise ValueError(f"Unknown command: {cmd}")

            args_dict = dict()
            for args_match in args_matcher.finditer(args):
                args_dict[args_match.group("key")] = args_match.group("value")

            self.ops.append(klass(name, **args_dict))

            if post:
                self.ops.append(Echo(post))

    def _run(self) -> None:
        for op in self.ops:
            if isinstance(op, Echo):
                self.draft += op.value
            else:
                self.output[op.name] = op.run(
                    self.tokenizer,
                    self.generator,
                    self.settings,
                    self.draft,
                )
                self.draft += self.output[op.name]
