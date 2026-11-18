"""One tool-use interface over Claude (Vertex AI) and Gemini.

A provider is a *stateful chat session*: it keeps its own native message history, because a lossy
neutral format drops things the APIs need back verbatim (Claude's thinking blocks, Gemini's thought
signatures). The agent loop only ever sees :class:`Turn` and sends back :class:`ToolResult`.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

RETRY_WAITS = (30, 60, 120, 240, 480, 600, 600, 600)  # seconds; ~45 min in all before giving up


def _transient(e: Exception) -> bool:
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    return code in (429, 500, 502, 503, 504, 529) or "RESOURCE_EXHAUSTED" in str(e)


def with_retry(fn: Callable[[], Any]) -> Any:
    """Retry quota (429) and server errors with backoff. p1's Gemini Pro run hit Vertex's
    tokens-per-minute quota and the unretried 429 killed the whole run."""
    for wait in (*RETRY_WAITS, None):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if wait is None or not _transient(e):
                raise
            print(f"provider: {type(e).__name__} {str(e)[:120]!r}; retrying in {wait}s", flush=True)
            time.sleep(wait)


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
    finish: str = ""  # provider stop reason, e.g. "MALFORMED_FUNCTION_CALL", "EMPTY", "end_turn"
    thinking: str = ""  # reasoning summary, when the provider returns one
    cache_read_tokens: int = 0  # input tokens served from the prompt cache (Claude)
    cache_write_tokens: int = 0  # input tokens written to the prompt cache (Claude)


class Provider(Protocol):
    name: str
    model: str

    def start(self, system: str, tools: list[ToolSpec]) -> None: ...
    def send(self, message: str | list[ToolResult]) -> Turn: ...


def gcp_credentials(project: str):
    """Application-default credentials with ``project`` as the quota project, so calls are billed and
    rate-limited against the project we name, not whatever quota project the local ADC file carries."""
    import google.auth

    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"],
                                   quota_project_id=project)
    return creds


class ClaudeVertex:
    """Claude through Google Cloud Vertex AI, billed to the GCP project (and its credits).

    Cost: an agent run re-sends its whole history every turn (~2-4M input tokens a run), so prompt
    caching does most of the saving. One explicit breakpoint on the frozen system prompt (tools render
    before it, so they are covered too) plus top-level automatic caching for the growing tail. TTL is the
    default 5 minutes: in p2, 97% of turns started under 5 min after the previous one (median 19 s), so
    the 1-hour TTL's 2x write price would cost more than the rare miss after a long ``generate``.

    Thinking: adaptive with ``display: "summarized"`` so the transcript (and the Studio) shows the agent's
    reasoning; thinking blocks are kept in the history unchanged, as the API requires. ``effort`` sets
    how hard the model thinks (Opus 5.5 defaults to ``medium``; agentic work is run at ``high``).
    """

    name = "claude-vertex"

    def __init__(self, model: str, *, project: str | None = None, region: str | None = None,
                 max_tokens: int = 32000, effort: str = "high"):
        from anthropic import AnthropicVertex

        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        project = project or os.environ["WSP_GCP_PROJECT"]
        self.client = AnthropicVertex(
            project_id=project, region=region or os.environ.get("WSP_VERTEX_REGION", "global"),
            credentials=gcp_credentials(project),
        )
        self.system: list[dict] = []
        self.tools: list[dict] = []
        self.messages: list[dict] = []

    def start(self, system: str, tools: list[ToolSpec]) -> None:
        # frozen for the whole run: any byte change here would invalidate the cache for every later turn
        self.system = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
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

        def call():
            # streamed: long thinking + tool input can outlast a non-streaming request's HTTP timeout
            with self.client.messages.stream(
                model=self.model, max_tokens=self.max_tokens, system=self.system, tools=self.tools,
                messages=self.messages, cache_control={"type": "ephemeral"},
                thinking={"type": "adaptive", "display": "summarized"},
                output_config={"effort": self.effort},
            ) as stream:
                return stream.get_final_message()

        resp = with_retry(call)
        u = resp.usage
        read = getattr(u, "cache_read_input_tokens", 0) or 0
        write = getattr(u, "cache_creation_input_tokens", 0) or 0
        stop = resp.stop_reason or ""
        if stop in ("refusal", "max_tokens") or not resp.content:
            # a refusal, or a reply cut off mid tool call: not kept; the agent loop re-prompts
            return Turn("", [], u.input_tokens, u.output_tokens, raw=resp, finish=stop or "EMPTY",
                        cache_read_tokens=read, cache_write_tokens=write)
        self.messages.append({"role": "assistant", "content": resp.content})  # thinking blocks unchanged
        text = "".join(b.text for b in resp.content if b.type == "text")
        thinking = "\n".join(b.thinking for b in resp.content if b.type == "thinking" and b.thinking)
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if b.type == "tool_use"]
        return Turn(text, calls, u.input_tokens, u.output_tokens, raw=resp, finish=stop, thinking=thinking,
                    cache_read_tokens=read, cache_write_tokens=write)


class Gemini:
    """Gemini through Vertex AI on ``WSP_GCP_PROJECT`` (GCP credits), or the Gemini API when
    ``GEMINI_API_KEY`` is set and no project is."""

    name = "gemini"

    def __init__(self, model: str, *, api_key: str | None = None, project: str | None = None,
                 location: str = "global", seed: int | None = None, include_thoughts: bool = True):
        from google import genai

        self.model = model
        self.seed = seed
        self.include_thoughts = include_thoughts
        project = project or os.environ.get("WSP_GCP_PROJECT")
        if project:
            self.client = genai.Client(vertexai=True, project=project, location=location,
                                       credentials=gcp_credentials(project))
        else:
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
            seed=self.seed,
            # thought summaries: the transcript and the Studio show the agent's reasoning, not only its code
            thinking_config=types.ThinkingConfig(include_thoughts=True) if self.include_thoughts else None,
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
        resp = with_retry(lambda: self.client.models.generate_content(
            model=self.model, contents=self.contents, config=self.config))
        u = resp.usage_metadata
        out = ((getattr(u, "candidates_token_count", 0) or 0) + (getattr(u, "thoughts_token_count", 0) or 0)) if u else 0
        inp = (getattr(u, "prompt_token_count", 0) or 0) if u else 0
        cand = resp.candidates[0] if resp.candidates else None
        finish = str(getattr(getattr(cand, "finish_reason", None), "name", "") or "") if cand else "EMPTY"
        content = cand.content if cand else None
        parts = (content.parts or []) if content else []
        if not parts:
            # MALFORMED_FUNCTION_CALL, SAFETY, an empty candidate...: nothing to keep in the history (an
            # empty model turn is rejected on the next call); the agent loop re-prompts.
            return Turn("", [], inp, out, raw=resp, finish=finish or "EMPTY")
        self.contents.append(content)  # kept whole: carries thought signatures back
        text = "".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False))
        thinking = "".join(p.text for p in parts if getattr(p, "text", None) and getattr(p, "thought", False))
        calls = [ToolCall(p.function_call.id or f"call_{i}", p.function_call.name,
                          dict(p.function_call.args or {}))
                 for i, p in enumerate(parts) if getattr(p, "function_call", None)]
        return Turn(text, calls, inp, out, raw=resp, finish=finish, thinking=thinking)


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
        return self.script.pop(0) if self.script else Turn("done", [], 0, 0, finish="end_turn")


def make_provider(kind: str, model: str, **kw: Any) -> Provider:
    if kind == "claude-vertex":
        return ClaudeVertex(model, **kw)
    if kind == "gemini":
        return Gemini(model, **kw)
    if kind == "fake":  # tests and dry runs: kw["script"] is a list of {"text", "calls": [[name, args]]}
        turns = [Turn(t.get("text", ""), [ToolCall(str(i), n, a) for i, (n, a) in enumerate(t.get("calls", []))],
                      t.get("input_tokens", 0), t.get("output_tokens", 0), finish=t.get("finish", ""))
                 for t in kw.get("script", [])]
        return FakeProvider(turns, model=model)
    raise ValueError(f"unknown provider {kind!r}")
