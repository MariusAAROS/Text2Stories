import subprocess
import sys

root = subprocess.run(["git", "rev-parse", "--show-toplevel"],
capture_output=True,
text=True,
encoding="utf-8").stdout.strip()

sys.path.append(root)

DEBUG = False

from langgraph.graph import MessagesState, StateGraph, START, END

from pydantic import Field
from typing import List, TypedDict, Annotated
from operator import add

from Text2Stories.utils.types import UserStory, Epic, FlatUserStory, Need
from Text2Stories.utils.llm import slow_mistral
from Text2Stories.models.teams.elicitation import elicitator_builder
from Text2Stories.models.teams.scheduling import scheduler_builder
from Text2Stories.models.teams.validation import validator_builder

llm = slow_mistral

class USGeneratorState(MessagesState):
    # Elicitator
    needs: Annotated[List[Need], add]
    summarized_needs: str
    feedback: str
    grade: str
    max_iter: int
    sum_iter: int
    current_iter: int
    min_needs: int
    product: str = Field("Product to be built")
    subject: str = Field("Subject of the project")
    task: str = Field("Task to be accomplished")

    # Scheduler
    epics: Annotated[List[Epic], add]
    completed_epics: Annotated[List[Epic], add]

    # Validator
    validated_epics: Annotated[List[Epic], add]
    user_stories: List[UserStory]
    validated_user_stories: Annotated[List[FlatUserStory], add]
    amigos_max_iter: int

class USGeneratorInputState(MessagesState):
    max_iter: int
    sum_iter: int
    min_needs: int
    amigos_max_iter: int
    product: str = Field("Product to be built")
    subject: str = Field("Subject of the project")
    task: str = Field("Task to be accomplished")

class USGeneratorOutputState(TypedDict):
    validated_epics: Annotated[List[Epic], add]

def initialize_state(state: USGeneratorState):
    return {
        "current_iter": 0
        }

def state_check(state: USGeneratorState):
    if not state.get("summarized_needs"):
        return {"summarized_needs": "No summary yet"}
    else:
        return {}

def state_cleanup(state: USGeneratorState):
    """Clean up the state"""
    for epic in state["completed_epics"]:
        # if 1 or less user stories , remove the epic
        if epic.user_stories and len(epic.user_stories) <= 1:
                state["completed_epics"].remove(epic)

us_generator_builder = StateGraph(
    USGeneratorState,
    input=USGeneratorInputState,
    output=USGeneratorOutputState,
)

us_generator_builder.add_node("Initialize", initialize_state)
us_generator_builder.add_node("Elicitator", elicitator_builder.compile())
us_generator_builder.add_node("Scheduler", scheduler_builder.compile())
us_generator_builder.add_node("Validator", validator_builder.compile())
us_generator_builder.add_node("State Integrity", state_check)
us_generator_builder.add_node("State Cleanup", state_cleanup)

us_generator_builder.add_edge(START, "Initialize")
us_generator_builder.add_edge("Initialize", "Elicitator")
us_generator_builder.add_edge("Elicitator", "State Integrity")
us_generator_builder.add_edge("State Integrity", "Scheduler")
us_generator_builder.add_edge("Scheduler", "State Cleanup")
us_generator_builder.add_edge("State Cleanup", "Validator")
us_generator_builder.add_edge("Validator", END)