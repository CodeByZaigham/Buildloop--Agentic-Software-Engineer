from .nodes import requirement_analysis,project_planner,architectural_node,API_designer,code_writer,compilation_checker,human_approval,deliver_project
from langgraph.graph.state import StateGraph,START,END
from langgraph.checkpoint.memory import MemorySaver
from .state import AgentState

graph=StateGraph(AgentState)

#routers
def compile_check(state:AgentState)->str:
     if(state["code_review"]["valid"]):
          return "human_approval"
     if(state["revision_count"] > 2):
          return "human_approval"
     return "code_writer"

def human_review_check(state:AgentState)->str:
     if state["human_approved"]:
          return "deliver_project"
     return "code_writer"

# def check_status(state:AgentState) -> str:
#      if state["status"]=="completed":
#           return END
#      return 

#Nodes
graph.add_node("analyzer",requirement_analysis)
graph.add_node("project_planner",project_planner)
graph.add_node("architectural_node",architectural_node)
graph.add_node("API_designer",API_designer)
graph.add_node("code_writer",code_writer)
graph.add_node("human_approval",human_approval)
graph.add_node("deliver_project",deliver_project)
graph.add_node("compilation_checker",compilation_checker)

#Edges
graph.add_edge(START, "analyzer")
graph.add_edge("analyzer","project_planner")
graph.add_edge("project_planner","architectural_node")
graph.add_edge("project_planner","API_designer")
graph.add_edge("architectural_node","code_writer")
graph.add_edge("API_designer","code_writer")
graph.add_edge("code_writer","compilation_checker")
graph.add_conditional_edges("compilation_checker",compile_check)
graph.add_conditional_edges("human_approval",human_review_check)
graph.add_edge("deliver_project",END)

checkpoint=MemorySaver()
app=graph.compile(checkpointer=checkpoint)