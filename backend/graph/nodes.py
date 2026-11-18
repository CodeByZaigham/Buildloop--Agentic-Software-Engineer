from .state import AgentState
import re
import json
from .llm import Groq, Mistral, parse_llm_json
from langgraph.types import interrupt

TOOLS_NOTE = """
    You have access to a web_search tool. Only call it when you genuinely
    need up-to-date, real-world information (e.g. verifying that a specific
    library/technology mentioned in the requirement still exists, or how it
    is currently used). Do not call it for well-known general knowledge,
    and never call it more than necessary.
    """

def _parse_json(content: str) -> dict:
     """Thin wrapper kept for readability at call sites - see llm.parse_llm_json
     for the actual (json_repair-backed) parsing logic."""
     return parse_llm_json(content)