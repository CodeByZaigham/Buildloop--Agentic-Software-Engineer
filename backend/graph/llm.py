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