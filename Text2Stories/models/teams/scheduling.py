import subprocess
import sys, os, yaml

root = subprocess.run(["git", "rev-parse", "--show-toplevel"],
capture_output=True,
text=True,
encoding="utf-8").stdout.strip()

sys.path.append(root)

DEBUG = False

from langgraph.graph import MessagesState, StateGraph, START, END
from pydantic import ValidationError

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.constants import Send

from typing import List, Annotated
from operator import add

from Text2Stories.utils.types import Epic, Epics, Need
from Text2Stories.utils.llm import slow_mistral

with open(os.path.join(root, "Text2Stories", "utils", "prompts.yaml"), "r") as f:
    prompts = yaml.safe_load(f)

llm = slow_mistral

class SchedulerState(MessagesState):
    needs: Annotated[List[Need], add]
    summarized_needs: str
    epics: Annotated[List[Epic], add]
    completed_epics: Annotated[List[Epic], add]

class SchedulerInputState(MessagesState):
    needs: Annotated[List[Need], add]
    summarized_needs: str

class SchedulerOutputState(MessagesState):
    completed_epics: Annotated[List[Epic], add]

class WorkerState(MessagesState):
    epic: Epic
    needs: Annotated[List[Need], add]
    completed_epics: Annotated[List[Epic], add]

llm_product_manager = llm.with_structured_output(Epics)
llm_product_owner = llm.with_structured_output(Epic)

def summarize_needs(state: SchedulerState):
    """Summarize the needs for the project"""

    # Summarize the needs
    try:
        summarized_needs = llm.invoke(
            [
                SystemMessage(content=prompts["Scheduling"]["summarize_sys"]),
                HumanMessage(content=prompts["Scheduling"]["summarize_human"].format(
                    needs=state["needs"],
                    summary=state["summarized_needs"] if state.get("summarized_needs") else "No summary"
                ))
            ]
        )
    except ValidationError as e:
        print("error in summarized_needs :", e)
        summarized_needs = ""

    return {"summarized_needs": summarized_needs}

# Nodes
def orchestrator(state: SchedulerState):
    """Orchestrator that generates epics for the project"""

    # Generate queries
    try:
        output = llm_product_manager.invoke(
            [
                SystemMessage(content=prompts["Scheduling"]["generate_epics_sys"]),
                HumanMessage(content=prompts["Scheduling"]["generate_epics_human"].format(
                    needs=state["needs"],
                    summarized_needs=state["summarized_needs"] if state.get("summarized_needs") else "No summary"
                )),
            ]
        )
    except ValidationError as e:
        print("error in orchestrator :", e)
        output = Epics(epics=[])

    return {"epics": output.epics}

def worker(state: WorkerState):
    """Worker writes an epic of the project"""

    # Generate section
    try:
        epic = llm_product_owner.invoke(
            [
                SystemMessage(
                    content=prompts["Scheduling"]["generate_user_stories_sys"]
                ),
                HumanMessage(
                    content=prompts["Scheduling"]["generate_user_stories_human"].format(
                        epic_desc=state["epic"].epic,
                        summarized_needs=state["summarized_needs"]
                    )
                ),
            ]
        )
    except ValidationError as e:
        print("error in worker :", e)
        epic = []

    # Write the updated section to completed sections
    return {"completed_epics": [epic]}

# Conditional edge function to create llm_call workers that each write a section of the report
def assign_workers(state: SchedulerState):
    """Assign a worker to each section in the plan"""

    # Kick off section writing in parallel via Send() API
    
    return [Send("Worker", {"epic": s,
                            "summarized_needs": state["summarized_needs"]}) for s in state["epics"]]

# Build workflow
scheduler_builder = StateGraph(SchedulerState,
                                         input=SchedulerInputState,
                                         output=SchedulerOutputState)

# Add the nodes
scheduler_builder.add_node("Summarizer", summarize_needs)
scheduler_builder.add_node("Orchestrator", orchestrator)
scheduler_builder.add_node("Worker", worker)

# Add edges to connect nodes
scheduler_builder.add_edge(START, "Summarizer")
scheduler_builder.add_edge("Summarizer", "Orchestrator")
scheduler_builder.add_conditional_edges(
    "Orchestrator", assign_workers, ["Worker"]
)
scheduler_builder.add_edge("Worker", END)