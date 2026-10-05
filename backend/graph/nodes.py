from .state import AgentState
from .llm import Groq, Mistral, invoke_resilient, invoke_with_tools, parse_llm_json
from langgraph.types import interrupt
from config import get_project_dir
from dispatcher.create_files import create_structure
from dispatcher.zip_folder import zip_project
from validator.validate import validate_project
import re
import json

# LLM providers are assigned per node with a primary + a fallback (see
# invoke_resilient in llm.py): every node's real workhorse is its primary
# provider, but if that provider can't deliver (rate limit exhausted, or a
# request that structurally doesn't fit its per-minute token budget), the
# call automatically retries against the other provider instead of failing
# the whole node.
#
# Assignment isn't just round-robin - it's weighted by how big each node's
# prompt tends to get. code_writer receives the full accumulated project
# context (and, on revision, complete file contents), making it by far the
# heaviest prompt in the pipeline - so it's pinned to Mistral as primary
# (Mistral's free tier allows far more tokens per minute than Groq's free
# tier does for this model). The lighter, earlier-stage nodes alternate
# Mistral/Groq to spread load. There is deliberately no AI code-review node
# any more (see the note above compilation_checker below) - that was the
# single heaviest, most rate-limit-prone call in the whole pipeline.

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

#To map code files into the dictionary.
def parse_files(content: str) -> dict[str, str]:
     pattern = r"<FILE>\s*(.*?)\s*<CONTENT>\s*(.*?)\s*</CONTENT>\s*</FILE>"

     matches = re.findall(
          pattern,
          content,
          re.DOTALL
     )

     files = {}

     for path, code in matches:
          files[path.strip()] = code

     return files

#NODES:

def requirement_analysis(state:AgentState) -> dict:
     """perform analysis and diversify user requirements"""
     requirement=state["requirement"]
     system_prompt="""
     You are a software requirements analyst.

     Your task is to analyze the user's software project requirement and transform it
     into a clear, structured specification for downstream AI agents.

     Identify:

     1. The main goal of the project.
     2. The core features that need to be implemented.
     3. The technologies explicitly requested by the user.
     4. Important constraints or requirements that the implementation must follow.

     Do not design the architecture, write code, or invent requirements that were not
     provided by the user.

     If some information is not specified, make a reasonable assumption only when
     necessary and clearly identify it as an assumption.
     """ + TOOLS_NOTE + """
     Return the result in the following structure:

     {
     "goal": "...",
     "features": ["...", "..."],
     "technologies": ["...", "..."],
     "constraints": ["...", "..."]
     }
     Do not wrap the response in Markdown code fences.
     """
     user_prompt=f"requirement: {requirement}"

     response=invoke_with_tools(
          Mistral(use_tools=True), [("system",system_prompt),("human",user_prompt)],
          fallback=Groq(use_tools=True),
     )
     result=_parse_json(response.content)
     return {"analysis":result}

def project_planner(state:AgentState) -> dict:
     """plans the project according to the requirements defined"""
     requirement=state["requirement"]
     analysis=state["analysis"]
     system_prompt="""
     You are an expert software project planner.

     Your task is to create a clear and practical implementation plan for a software
     project based on the user's original requirement and the requirement analysis.

     The plan will be passed to downstream agents responsible for architecture,
     API design, and code generation.

     Determine:

     1. The type of software project.
     2. The major components/modules required.
     3. The files that should be created.
     4. The dependencies/libraries required.

     Rules:
     - Follow the user's stated requirements and technologies.
     - Use the requirement analysis as the primary guide.
     - Do not write implementation code.
     - Do not design detailed architecture.
     - Do not invent unnecessary features.
     - Do not define files give only their names/paths.
     - Keep the project structure practical and maintainable.
     - Only include dependencies that are actually needed.
     """ + TOOLS_NOTE + """
     Return the result in this structure:

     {
     "project_type": "...",
     "components": [
          "...",
          "..."
     ],
     "files": [
          "...",
          "..."
     ],
     "dependencies": [
          "...",
          "..."
     ]
     }
     Do not wrap the response in Markdown code fences.
     """
     user_prompt =f"""
     Original requirement:

     {requirement}

     Requirement analysis:

     {analysis}

     Create an implementation plan based on the information above.
     """
     response=invoke_with_tools(
          Groq(use_tools=True), [("system",system_prompt),("human",user_prompt)],
          fallback=Mistral(use_tools=True),
     )
     result=_parse_json(response.content)
     return {"project_plan":result}

def architectural_node(state: AgentState) -> dict:
     "this node reads the requirement analysis and project planings and creates a architecture of it."
     analysis=state["analysis"]
     planning=state["project_plan"]
     system_prompt="""
     You are a senior software architect.

     Your task is to design a practical software architecture based on the
     requirement analysis and project plan provided to you.

     Define how the planned components should be organized and how data should
     flow through the system.

     Determine:

     1. The overall architectural pattern.
     2. The major components and their responsibilities.
     3. How the components interact with each other.
     4. The application's high-level data flow.

     Rules:
     - Follow the requirements and project plan.
     - Do not add unnecessary features or components.
     - Do not write implementation code.
     - Do not define detailed API endpoints; that will be handled by a separate API
     design agent.
     - Keep the architecture simple, maintainable, and appropriate for the project.
     - Clearly explain the responsibility of each component.
     - The architecture must be directly usable by the code generation agent.

     Return the result in this structure:

     {
     "pattern": "...",
     "components": {
          "component_name": "responsibility",
          "component_name": "responsibility"
     },
     "data_flow": [
          "Step 1",
          "Step 2",
          "Step 3"
     ]
     }
     Do not wrap the response in Markdown code fences.
     """
     user_prompt=f"""
     Design the architecture for the following software project.

     Requirement Analysis:
     {analysis}

     Project Plan:
     {planning}

     Create the high-level architecture based strictly on 
     the information above.
     """
     response=invoke_resilient(Mistral(), Groq(), [("system",system_prompt),("human",user_prompt)])
     result=_parse_json(response.content)
     return {"architecture":result}

def API_designer(state:AgentState) -> dict:
     "this node reads requirement analysis and project planning and design api endpoints and their purpose"
     analysis=state["analysis"]
     planning=state["project_plan"]
     system_prompt="""
     You are a senior API designer.

     Your task is to design the API contract for a software project based on the
     requirement analysis and project plan provided by the user.

     Define the API endpoints and the data models required to communicate with
     the application.

     Determine:

     1. The HTTP endpoints required by the project's features.
     2. The HTTP method for each endpoint.
     3. The purpose of each endpoint.
     4. The request models required by the API.
     5. The response models returned by the API.

     Rules:
     - Design only the APIs required by the given requirements.
     - Follow the project plan and identified features.
     - Do not invent unnecessary endpoints or features.
     - Do not write implementation code.
     - Do not decide the internal architecture; that is handled by the architecture
     node.
     - Use appropriate REST conventions.
     - Keep the API simple and consistent.
     - Make the output detailed enough for a code generation node to implement.

     Return the result in this structure:

     {
     "endpoints": [
          {
               "method": "...",
               "path": "...",
               "purpose": "..."
          }
     ],
     "request_models": [
          {
               "name": "...",
               "fields": ["...", "..."]
          }
     ],
     "response_models": [
          {
               "name": "...",
               "fields": ["...", "..."]
          }
     ]
     }
     Do not wrap the response in Markdown code fences.
     """
     user_prompt=f"""
     Design the API contract for the following software project.

     Requirement Analysis:
     {analysis}

     Project Plan:
     {planning}

     Based strictly on the information above, define the required API endpoints,
     request models, and response models.
     """
     response=invoke_resilient(Groq(), Mistral(), [("system",system_prompt),("human",user_prompt)])
     result=_parse_json(response.content)
     return {"api_design":result}

def code_writer(state:AgentState) -> dict:
     analysis=state["analysis"]
     planning=state["project_plan"]
     architecture=state["architecture"]
     api_design=state["api_design"]
     Regeneration_attempts=state["revision_count"] + 1

     system_prompt=f"""
     You are an expert software engineer responsible for implementing a software
     project from its requirements, project plan, architecture, and API design.

     Your task is to generate the complete project source code based on the
     information provided.

     The previous nodes have already determined the project's requirements,
     components, architecture, files, dependencies, and API.

     Your job is ONLY to implement that specification.

     Do not introduce new features, files, dependencies, architectural patterns,
     API endpoints, database technologies, or project components that are not
     present in the provided specification.

     If something is not specified, choose the simplest implementation necessary
     to satisfy the existing specification.

     Follow these rules:

     1. Implement only the requirements defined in the project specification.
     2. Follow the provided project architecture.
     3. Follow the API contract exactly.
     4. Use the planned project structure and dependencies.
     5. Write clean, readable, maintainable, production-quality code.
     6. Keep responsibilities separated according to the architecture.
     7. Do not add unnecessary features or technologies.
     8. Ensure that files work together as a coherent project.
     9. Include all necessary imports and basic error handling.
     10. When revising existing code, preserve working functionality and modify
     only what is necessary to address the provided feedback.

     Return the project as a collection of files.

     For every file, return:
     - the file path
     - the complete file content

     Use exactly this structure:

     <FILE>
     FILE_PATH
     <CONTENT>
     COMPLETE FILE CONTENT
     </CONTENT>
     </FILE>

     Example:

     <FILE>
     app/main.py
     <CONTENT>
     from fastapi import FastAPI

     app = FastAPI()
     </CONTENT>
     </FILE>

     Do not wrap the response in Markdown code fences.
     Do not return explanations outside this structure.
     """

     if Regeneration_attempts<1:
          user_prompt=f"""
          Generate the complete implementation for the following software project.

          REQUIREMENT ANALYSIS:
          {analysis}

          PROJECT PLAN:
          {planning}

          ARCHITECTURE:
          {architecture}

          API DESIGN:
          {api_design}

          Implement the project according to all of the specifications above.

          Generate every required project file with complete, executable source code.
          """

     else:
          #this both will be used when regeneration needed or when attempt is greater than 1
          current_code=state["files"]
          compilation_review=state["code_review"]
          human_feedback=state["human_feedback"]
          user_prompt=f"""
          Revise the existing project implementation based on the feedback provided
          below.

          REQUIREMENT ANALYSIS:
          {analysis}

          PROJECT PLAN:
          {planning}

          ARCHITECTURE:
          {architecture}

          API DESIGN:
          {api_design}

          CURRENT PROJECT FILES:
          {current_code}

          COMPILATION / LINT CHECK:
          {compilation_review}

          HUMAN FEEDBACK:
          {human_feedback}

          Your task is to produce the corrected version of the project.

          Requirements for revision:

          1. Fix every issue identified in the compilation/lint check.
          2. Address the human feedback if provided.
          3. Preserve functionality that is already correct.
          4. Do not introduce unnecessary changes.
          5. Keep the implementation consistent with the architecture and API design.
          6. Return the complete updated contents of every project file, not only the
          files that were changed.

          Return the result using the same file structure specified in the system prompt.
          """
     # Heaviest prompt in the pipeline (full file contents on revision) -
     # Mistral primary, since its free-tier token budget is far larger than
     # Groq's for this model. Groq is kept as an emergency fallback only.
     response=invoke_resilient(Mistral(), Groq(), [("system",system_prompt),("human",user_prompt)])
     return {"files":parse_files(response.content),"revision_count":Regeneration_attempts}

# NOTE: the AI code_reviewer node was removed intentionally.
#
# It was the single heaviest prompt in the pipeline (full project source +
# every prior stage's output) and, on both Groq's and Mistral's free tiers,
# was the node most likely to blow through per-minute token budgets -
# causing the exact "context window"/rate-limit failures this pipeline used
# to hit. Removing it cuts LLM calls per run from 6 to 5 and removes the
# most fragile one.
#
# Review duties are now split between:
#   - compilation_checker: deterministic Ruff + py_compile check, runs with
#     zero LLM calls and auto-fixes what it safely can (unused imports) via
#     `ruff check --fix`.
#   - human_approval: a real person makes the final call, informed by the
#     compilation/lint results.

def compilation_checker(state:AgentState)->dict:
     """
     Compiles and lints the generated project (no LLM call). Ruff
     auto-fixes what it safely can (currently: unused imports) and the
     corrected file contents are written back into state - so a trivial
     fix doesn't cost a code_writer round trip. Anything Ruff can't safely
     auto-fix (e.g. bare/blind except) is reported and, per compile_check
     in edges.py, sent back to code_writer for a real revision.
     """
     files=state["files"]
     response=validate_project(files)
     return {
          "code_review":{
               "valid":response["valid"],
               "syntax_errors":response["syntax_errors"],
               "unresolved_imports":response["unresolved_imports"],
               "lint_issues":response["lint_issues"],
          },
          "files":response["files"],
     }

def human_approval(state:AgentState)->dict:
     """this node takes human approval before creating final folders"""
     human_response=interrupt({
          "Files":list(state["files"].keys()),
          "Compiler_check":state["code_review"],
          "instructions":"Type accept to approve or type your feedback to request another revision"
     })

     if human_response.strip().lower() in ["approved","approve","good","yes","go ahead"]:
          return {"human_approved":True}
     return {"human_feedback":human_response}

def deliver_project(state:AgentState):
     """this reads the files dictionary and writes the project into its own
     folder under config.BASE_DIR, zips it, then deletes the raw folder -
     leaving only "<BASE_DIR>/<project_id>.zip" behind."""
     files=state["files"]
     project_dir=get_project_dir(state["project_id"])
     status=create_structure(files, base_dir=str(project_dir))
     if status:
          zip_path=zip_project(base_dir=str(project_dir))
          return {"deliverable":zip_path , "status":"completed"}
     return {"status":"error"}
