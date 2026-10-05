import json
import random
import time

import httpx
import json_repair
from groq import RateLimitError as GroqRateLimitError
from langchain_core.messages import ToolMessage
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

from config import settings
from tools.web_search import search_tool

load_dotenv()

# Tools are only ever bound onto the analyzer/planner LLMs (see nodes.py) -
# every other node gets a plain, tool-free model.
TOOLS = [search_tool]
TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}


def Groq(use_tools: bool = False) -> ChatGroq:
     llm = ChatGroq(model="openai/gpt-oss-120b")
     return llm.bind_tools(TOOLS) if use_tools else llm

def Mistral(use_tools: bool = False) -> ChatMistralAI:
     llm = ChatGroq(model="openai/gpt-oss-120b")
     return llm.bind_tools(TOOLS) if use_tools else llm


# ---------------------------------------------------------------------------
# Robust JSON parsing
# ---------------------------------------------------------------------------
# Models routinely wrap their JSON in prose ("Here's the analysis: {...}"),
# Markdown fences, trailing commas, or cut it off mid-object when they hit a
# token limit. json.loads chokes on all of that, so we try it as a fast path
# for well-formed output and fall back to json_repair (which is specifically
# built to recover from exactly these LLM-shaped mistakes) otherwise.

def parse_llm_json(content: str) -> dict:
     """
     Parse a JSON object out of a raw LLM response, tolerating leading/
     trailing prose, Markdown code fences (```json ... ``` or ``` ... ```
     anywhere in the text), trailing commas, single quotes, and truncated
     JSON. Raises ValueError (with a snippet of the offending text) if
     nothing usable can be recovered.
     """
     if not content or not content.strip():
          raise ValueError("Model returned an empty response - nothing to parse as JSON.")

     text = content.strip()

     # Fast path: the model behaved and gave us clean JSON.
     try:
          return json.loads(text)
     except json.JSONDecodeError:
          pass

     # Slow path: recover from fences / stray prose / minor syntax mistakes.
     try:
          repaired = json_repair.loads(text)
     except Exception as exc:
          raise ValueError(
               f"Could not parse a JSON object out of the model's response. "
               f"First 300 chars:\n{text[:300]!r}"
          ) from exc

     if isinstance(repaired, dict):
          return repaired

     raise ValueError(
          f"Model's response parsed as {type(repaired).__name__}, not a JSON object. "
          f"First 300 chars:\n{text[:300]!r}"
     )


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------
# Different failure modes need different responses:
#   - rate_limit  -> transient, worth waiting out and retrying
#   - too_large   -> the request itself will never fit (e.g. Groq's free-tier
#                    per-minute token cap, or a genuine context-length
#                    overflow) - retrying the SAME provider is pointless,
#                    but a different provider with more headroom might work
#   - transient   -> 5xx / connection hiccups, worth a short retry
#   - fatal       -> a real bug (bad request, auth, etc.) - surface it as-is

_TOO_LARGE_HINTS = (
     "context_length_exceeded",
     "context length",
     "maximum context length",
     "reduce the length",
     "reduce your message size",
     "request too large",
     "too many tokens",
     "input is too long",
     "prompt is too long",
)


def _status_code(exc: Exception):
     response = getattr(exc, "response", None)
     if response is not None:
          return getattr(response, "status_code", None)
     return getattr(exc, "status_code", None)


def _error_text(exc: Exception) -> str:
     """Best-effort extraction of everything that might describe the error,
     lower-cased, for keyword matching."""
     parts = [str(exc)]
     response = getattr(exc, "response", None)
     if response is not None:
          try:
               parts.append(response.text)
          except Exception:
               pass
     body = getattr(exc, "body", None)
     if body:
          parts.append(str(body))
     return " ".join(parts).lower()


def classify_llm_error(exc: Exception) -> str:
     status = _status_code(exc)
     text = _error_text(exc)

     if status == 413 or any(hint in text for hint in _TOO_LARGE_HINTS):
          return "too_large"
     if status == 429 or isinstance(exc, GroqRateLimitError):
          return "rate_limit"
     if status is not None and status >= 500:
          return "transient"
     if isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError)):
          return "transient"
     return "fatal"


class LLMUnavailableError(RuntimeError):
     """Raised when an LLM call fails in a way retries on that same
     provider can't fix (too_large), or after every retry has been
     exhausted (rate_limit / transient). Callers can catch this to fall
     back to a different provider - see invoke_resilient."""


# ---------------------------------------------------------------------------
# Retry (rate-limit + transient errors only) and cross-provider fallback
# ---------------------------------------------------------------------------
# Retry count / base backoff are controlled by LLM_MAX_RETRIES and
# LLM_RETRY_BASE_DELAY in config.py (env-overridable).

def _retry_wait_seconds(exc: Exception):
     """Returns the API's suggested wait time (Retry-After), if it sent one."""
     response = getattr(exc, "response", None)
     retry_after = response.headers.get("retry-after") if response is not None else None
     try:
          return float(retry_after) if retry_after is not None else None
     except ValueError:
          return None


def invoke_with_retry(llm, messages: list):
     """
     Invoke a chat model, retrying only on rate-limit / transient (5xx,
     connection) errors, up to LLM_MAX_RETRIES times, with exponential
     backoff (honoring the API's Retry-After header when it sends one).

     A "too_large" error (the request structurally can't fit - e.g. Groq's
     free-tier per-minute token cap, or a genuine context-length overflow)
     is NOT retried - retrying the identical payload against the same
     provider would just fail again. It's raised immediately as
     LLMUnavailableError so a caller using invoke_resilient can fail over
     to a different provider instead. Any other ("fatal") error is raised
     as-is, unmodified - this is a resilience net, not a way to mask bugs.
     """
     attempt = 0
     model_name = getattr(llm, "model", getattr(llm, "model_name", "llm"))

     while True:
          try:
               return llm.invoke(messages)
          except Exception as exc:
               kind = classify_llm_error(exc)

               if kind == "too_large":
                    raise LLMUnavailableError(
                         f"{model_name}: request doesn't fit this model/tier's token budget ({exc})"
                    ) from exc

               if kind not in ("rate_limit", "transient"):
                    raise  # fatal / unknown - don't retry, don't mask it

               if attempt >= settings.LLM_MAX_RETRIES:
                    raise LLMUnavailableError(
                         f"{model_name}: still failing ({kind}) after {attempt} retries ({exc})"
                    ) from exc

               wait = _retry_wait_seconds(exc) if kind == "rate_limit" else None
               delay = wait if wait else settings.LLM_RETRY_BASE_DELAY * (2 ** attempt)
               delay += random.uniform(0, 0.5)  # jitter
               attempt += 1
               print(
                    f"[{kind}] {model_name} failed. Retrying in {delay:.1f}s "
                    f"(attempt {attempt}/{settings.LLM_MAX_RETRIES})"
               )
               time.sleep(delay)


def invoke_resilient(primary, fallback, messages: list):
     """
     Try `primary` (with its own full retry budget via invoke_with_retry).
     If it ultimately can't deliver - retries exhausted, or a too_large
     error that retries can't fix anyway - automatically fall back to
     `fallback` (with its own fresh retry budget) instead of failing the
     whole node.
     """
     try:
          return invoke_with_retry(primary, messages)
     except LLMUnavailableError as primary_error:
          fallback_name = getattr(fallback, "model", getattr(fallback, "model_name", "fallback llm"))
          print(f"[fallback] primary LLM unavailable ({primary_error}); retrying with {fallback_name}")
          try:
               return invoke_with_retry(fallback, messages)
          except LLMUnavailableError as fallback_error:
               raise LLMUnavailableError(
                    f"Both providers failed. Primary: {primary_error} | Fallback: {fallback_error}"
               ) from fallback_error


# ---------------------------------------------------------------------------
# Tool-calling loop (used only by the analyzer & planner nodes)
# ---------------------------------------------------------------------------

def invoke_with_tools(primary, messages: list, fallback=None, max_tool_iterations: int = 3):
     """
     Invoke a tool-bound chat model, executing any web_search tool calls it
     makes and feeding the results back in, until it returns a final,
     tool-free answer (or max_tool_iterations is hit). If `fallback` is
     given, each step is resilient (see invoke_resilient) instead of tied
     to a single provider.
     """
     conversation = list(messages)

     def _call(msgs):
          if fallback is not None:
               return invoke_resilient(primary, fallback, msgs)
          return invoke_with_retry(primary, msgs)

     for _ in range(max_tool_iterations):
          response = _call(conversation)

          if not getattr(response, "tool_calls", None):
               return response

          conversation.append(response)
          for call in response.tool_calls:
               tool = TOOLS_BY_NAME.get(call["name"])
               try:
                    result = tool.invoke(call["args"]) if tool else f"Unknown tool: {call['name']}"
               except Exception as exc:
                    result = f"Tool '{call['name']}' failed: {exc}"
               conversation.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

     # Model kept calling tools past the budget - force a final, tool-free answer.
     conversation.append(("human", "Stop calling tools now and return only the final JSON answer."))
     return _call(conversation)
