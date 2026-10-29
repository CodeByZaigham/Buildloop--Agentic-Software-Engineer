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