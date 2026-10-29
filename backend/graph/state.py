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