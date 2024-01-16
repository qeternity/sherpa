import sys
import re

from typing import Any, Dict, Optional
from aphrodite import SamplingParams
from aphrodite.common.utils import Counter

request_counter = Counter()

# returns groups for cmd, name and args
op_matcher = re.compile(
    r"""(?P<pre>[^\{]*){{\s*(?P<cmd>[a-z]+)\s*\"(?P<name>\w+)\"(?P<args>[^\}]*)}}(?P<post>[^\{]*)"""
)
args_matcher = re.compile(r"""\s*(?P<key>\w+)=\"(?P<value>[^"]+)\"""")


class Context:
    def __init__(self, engine, prompt: str) -> None:
        self.prompt = prompt
        self.engine = engine

        self.draft = ""
        self.vars: Dict[str, Any] = dict()

        self.token_count = 0


class Op:
    pass


class NamedOp(Op):
    NULL = "null"

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def _return_null(self, context: Context) -> Context:
        context.draft += self.NULL
        context.vars[self.name] = None
        return context


class Echo(Op):
    def __init__(self, value: str) -> None:
        super().__init__()
        self.value = value

    def __repr__(self) -> str:
        return f"""Echo(value="{self.value.encode("unicode_escape").decode()}")"""

    async def run(self, context: Context) -> Context:
        context.draft += self.value
        return context


class Generate(NamedOp):
    def __init__(
        self,
        name: str,
        max_tokens: Optional[str] = None,
        regex: Optional[str] = None,
        stop_regex: Optional[str] = None,
        depends: Optional[str] = None,
    ) -> None:
        super().__init__(name)
        self.max_tokens = int(max_tokens.strip()) if max_tokens else None
        self.stop_regex = stop_regex
        if regex:
            assert stop_regex is not None
            self.regex = rf"({regex}|[nul]+)({stop_regex}.*)?" if regex else None
        else:
            self.regex = None
        self.depends = depends

    def __repr__(self) -> str:
        return f"Generate(name={self.name}, max_tokens={self.max_tokens}, stop_regex={self.stop_regex})"

    async def run(self, context: Context) -> Context:
        if self.depends and context.vars.get(self.depends) is None:
            return self._return_null(context)

        if self.regex:
            raise NotImplementedError("Regex not implemented")
        
        request_id = str(next(request_counter))
        sampling_params = SamplingParams(temperature=0.0, max_tokens=self.max_tokens or 1024)

        cnt = 0
        draft = ""
        async for resp in context.engine.generate(context.draft, sampling_params, request_id):
            draft = resp.outputs[0].text
            if self.stop_regex and re.search(self.stop_regex, draft):
                await context.engine.abort(request_id)
                break

        if self.stop_regex and re.search(self.stop_regex, draft):
            draft = re.split(self.stop_regex, draft, maxsplit=1)[0]

        context.draft += draft
        context.vars[self.name] = draft if draft != self.NULL else None
        context.token_count += cnt
        return context


class Prompt:
    def __init__(self, engine, prompt: str) -> None:
        self.context = Context(engine, prompt)
        self.ops = list()

    async def __call__(self) -> Context:
        self._prepare()
        await self._run()
        return self.context

    def _prepare(self) -> None:
        for op_match in op_matcher.finditer(self.context.prompt):
            pre = op_match.group("pre")
            cmd = op_match.group("cmd")
            name = op_match.group("name")
            args = op_match.group("args")
            post = op_match.group("post")

            if pre:
                self.ops.append(Echo(pre))

            if cmd == "gen":
                klass = Generate
            elif cmd == "select":
                raise NotImplementedError("Select not implemented")
            else:
                raise ValueError(f"Unknown command: {cmd}")

            args_dict = dict()
            for args_match in args_matcher.finditer(args):
                args_dict[args_match.group("key")] = args_match.group("value")

            self.ops.append(klass(name, **args_dict))

            if post:
                self.ops.append(Echo(post))

    async def _run(self) -> None:
        for op in self.ops:
            self.context = await op.run(self.context)
