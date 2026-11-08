"""One tool-use interface over Claude (Vertex AI) and Gemini.

A provider is a *stateful chat session*: it keeps its own native message history, because a lossy
neutral format drops things the APIs need back verbatim (Claude's thinking blocks, Gemini's thought
signatures). The agent loop only ever sees :class:`Turn` and sends back :class:`ToolResult`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict  # JSON schema for the arguments object


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class ToolResult:
    call_id: str
    name: str
    content: str


@dataclass
class Turn:
    text: str
    tool_calls: list[ToolCall]
    input_tokens: int
    output_tokens: int
    raw: Any = field(default=None, repr=False)


class Provider(Protocol):
    name: str
    model: str

    def start(self, system: str, tools: list[ToolSpec]) -> None: ...
    def send(self, message: str | list[ToolResult]) -> Turn: ...


class ClaudeVertex:
    """Claude through Google Cloud Vertex AI, billed to the GCP project (and its credits)."""

    name = "claude-vertex"

    def __init__(self, model: str, *, project: str | None = None, region: str | None = None,
                 max_tokens: int = 16000):
        from anthropic import AnthropicVertex

        self.model = model
        self.max_tokens = max_tokens
        self.client = AnthropicVertex(
            project_id=project or os.environ["WSP_GCP_PROJECT"],
            region=region or os.environ.get("WSP_VERTEX_REGION", "global"),
        )
        self.system = ""
        self.tools: list[dict] = []
        self.messages: list[dict] = []

    def start(self, system: str, tools: list[ToolSpec]) -> None:
        self.system = system
        self.tools = [{"name": t.name, "description": t.description, "input_schema": t.parameters}
                      for t in tools]
        self.messages = []

    def send(self, message: str | list[ToolResult]) -> Turn:
        if isinstance(message, str):
            self.messages.append({"role": "user", "content": message})
        else:
            self.messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": r.call_id, "content": r.content} for r in message
            ]})
        resp = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=self.system,
            tools=self.tools, messages=self.messages,
        )
        self.messages.append({"role": "assistant", "content": resp.content})
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if b.type == "tool_use"]
        return Turn(text, calls, resp.usage.input_tokens, resp.usage.output_tokens, raw=resp)


class Gemini:
    """Gemini through the Gemini API (``GEMINI_API_KEY``)."""

    name = "gemini"

    def __init__(self, model: str, *, api_key: str | None = None):
        from google import genai

        self.model = model
        self.client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])
        self.config: Any = None
        self.contents: list[Any] = []

    def start(self, system: str, tools: list[ToolSpec]) -> None:
        from google.genai import types

        decls = [types.FunctionDeclaration(name=t.name, description=t.description,
                                           parameters_json_schema=t.parameters) for t in tools]
        self.config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=decls)] if decls else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        self.contents = []

    def send(self, message: str | list[ToolResult]) -> Turn:
        from google.genai import types

        if isinstance(message, str):
            self.contents.append(types.Content(role="user", parts=[types.Part.from_text(text=message)]))
        else:
            self.contents.append(types.Content(role="user", parts=[
                types.Part.from_function_response(name=r.name, response={"result": r.content})
                for r in message
            ]))
        resp = self.client.models.generate_content(model=self.model, contents=self.contents,
                                                   config=self.config)
        content = resp.candidates[0].content
        self.contents.append(content)  # kept whole: carries thought signatures back
        parts = content.parts or []
        text = "".join(p.text for p in parts if getattr(p, "text", None))
        calls = [ToolCall(p.function_call.id or f"call_{i}", p.function_call.name,
                          dict(p.function_call.args or {}))
                 for i, p in enumerate(parts) if getattr(p, "function_call", None)]
        u = resp.usage_metadata
        out = (u.candidates_token_count or 0) + (getattr(u, "thoughts_token_count", 0) or 0)
        return Turn(text, calls, u.prompt_token_count or 0, out, raw=resp)


class FakeProvider:
    """Scripted provider for tests and dry runs: replays a list of Turns, costs nothing."""

    name = "fake"

    def __init__(self, script: list[Turn], model: str = "fake"):
        self.model = model
        self.script = list(script)
        self.received: list[str | list[ToolResult]] = []

    def start(self, system: str, tools: list[ToolSpec]) -> None:
        self.received = []

    def send(self, message: str | list[ToolResult]) -> Turn:
        self.received.append(message)
        return self.script.pop(0) if self.script else Turn("done", [], 0, 0)


def make_provider(kind: str, model: str, **kw: Any) -> Provider:
    if kind == "claude-vertex":
        return ClaudeVertex(model, **kw)
    if kind == "gemini":
        return Gemini(model, **kw)
    if kind == "fake":  # tests and dry runs: kw["script"] is a list of {"text", "calls": [[name, args]]}
        turns = [Turn(t.get("text", ""), [ToolCall(str(i), n, a) for i, (n, a) in enumerate(t.get("calls", []))],
                      0, 0) for t in kw.get("script", [])]
        return FakeProvider(turns, model=model)
    raise ValueError(f"unknown provider {kind!r}")
