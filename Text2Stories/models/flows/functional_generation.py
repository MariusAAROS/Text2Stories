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
    topic: str = Field("Topic of the project")

class USGeneratorInputState(MessagesState):
    topic: str = Field("Topic of the project")

class USGeneratorOutputState(TypedDict):
    backlog: Annotated[List[UserStory], add]

functional_generation_builder = StateGraph(
    USGeneratorState,
    input=USGeneratorInputState,
    output=USGeneratorOutputState
)

def qa(state: USGeneratorState) -> USGeneratorState:
    pass

def dev(state: USGeneratorState) -> USGeneratorState:
    pass

def po(state: USGeneratorState) -> USGeneratorState:
    pass

def expert(state: USGeneratorState) -> USGeneratorState:
    pass

def manager(state: USGeneratorState) -> USGeneratorState:
    pass

def human(state: USGeneratorState) -> USGeneratorState:
    pass

def route_from_manager(state:USGeneratorState) -> USGeneratorState:
    pass

def route_from_expert(state:USGeneratorState) -> USGeneratorState:
    pass

def route_from_po(state:USGeneratorState) -> USGeneratorState:
    pass

functional_generation_builder.add_node("QA", qa)
functional_generation_builder.add_node("DEV", dev)
functional_generation_builder.add_node("PO", po)
functional_generation_builder.add_node("Expert", expert)
functional_generation_builder.add_node("Manager", manager)
functional_generation_builder.add_node("Human", human)

#Manager links
functional_generation_builder.add_edge(START, "Manager")
functional_generation_builder.add_edge("Manager", END)
functional_generation_builder.add_conditional_edges("Manager",
                                                    route_from_manager,
                                                    {"PO": "PO",
                                                     "Expert": "Expert",
                                                     "Human": "Human"})

#Expert links
functional_generation_builder.add_conditional_edges("Expert",
                                                    route_from_expert,
                                                    {"Manager": "Manager",
                                                     "PO": "PO"})

#PO links
functional_generation_builder.add_conditional_edges("PO",
                                                    route_from_po,
                                                    {"Manager": "Manager",
                                                     "Expert": "Expert",
                                                     "QA": "QA",
                                                     "DEV": "DEV"})
                                                     
#QA links
functional_generation_builder.add_edge("QA", "PO")

#DEV links
functional_generation_builder.add_edge("DEV", "PO")

#Human links
functional_generation_builder.add_edge("Human", "Manager")