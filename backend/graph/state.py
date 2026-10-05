from typing import TypedDict,Literal
# from langgraph.graph.message import add_messages


class RequirementAnalysis(TypedDict):
    goal: str
    features: list[str]
    technologies: list[str]
    constraints: list[str]


class ProjectPlan(TypedDict):
    project_type: str
    components: list[str]
    files: list[str]
    dependencies: list[str]


class Architecture(TypedDict):
    pattern: str
    components: dict[str, str]
    data_flow: list[str]


class APIModel(TypedDict):
    name: str
    fields: list[str]


class APIDesign(TypedDict):
    endpoints: list[dict[str, str]]
    request_models: list[APIModel]
    response_models: list[APIModel]


class Compilation_Check(TypedDict):
    valid:bool
    syntax_errors:list[dict]
    unresolved_imports:list[dict]
    lint_issues:list[dict]




class AgentState(TypedDict):

    # =========================
    # USER INPUT
    # =========================

    requirement: tuple

    # id of this generation run; used to namespace its folder/zip under
    # config.BASE_DIR (see config.get_project_dir)
    project_id: str


    # =========================
    # REQUIREMENT ANALYSIS
    # =========================

    analysis: RequirementAnalysis


    # =========================
    # PROJECT PLANNING
    # =========================

    project_plan: ProjectPlan


    # =========================
    # DESIGN
    # =========================

    architecture: Architecture

    api_design: APIDesign


    # =========================
    # GENERATED CODE
    # =========================

    files: dict[str, str]


    # =========================
    # CODE REVIEW
    # =========================

    code_review: Compilation_Check
    


    # =========================
    # ITERATION
    # =========================

    revision_count: int


    # =========================
    # HUMAN-IN-THE-LOOP
    # =========================

    human_feedback: str

    human_approved: bool


    # =========================
    # WORKFLOW
    # =========================

    status: str
    
    deliverable:str


