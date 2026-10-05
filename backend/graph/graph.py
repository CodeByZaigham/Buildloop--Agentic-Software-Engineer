import uuid

from .edges import app
from langgraph.types import Command


project_id = uuid.uuid4().hex
config={"configurable":{"thread_id":project_id}}

state = {
    "requirement": (
        "Build a REST API for a Todo application "
        "using FastAPI and PostgreSQL."
    ),

    "project_id": project_id,

    "analysis": {
        "goal": "",
        "features": [],
        "technologies": [],
        "constraints": []
    },

    "project_plan": {
        "project_type": "",
        "components": [],
        "files": [],
        "dependencies": []
    },

    "architecture": {
        "pattern": "",
        "components": {},
        "data_flow": []
    },

    "api_design": {
        "endpoints": [],
        "request_models": [],
        "response_models": []
    },

    "files": {},

    "code_review":{
        "valid":False,
        "syntax_errors":[],
        "unresolved_imports":[],
        "lint_issues":[]
    },

    "revision_count": -1,

    "human_feedback": "",

    "human_approved": False,

    "status": "started"
}

result=app.invoke(state,config=config)

if "__interrupt__" in result:
    data=result["__interrupt__"][0].value
    print(data)
    response=input("\n approve or give feedback: ")
    result=app.invoke(Command(resume=response),config=config)

if result["status"]=="completed":
    print(result["deliverable"])
else:
    print(result["status"])
