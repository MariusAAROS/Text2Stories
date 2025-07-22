import subprocess
import sys

root = subprocess.run(["git", "rev-parse", "--show-toplevel"],
capture_output=True,
text=True,
encoding="utf-8").stdout.strip()

sys.path.append(root)

DEBUG = False

from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.constants import Send

from typing import List, Annotated
from operator import add

from pydantic import BaseModel

from Text2Stories.utils.types import UserStory, Epic, FlatUserStory
from Text2Stories.models.teams.amigos import amigos_builder, ThreeAmigosInputState
from Text2Stories.utils.llm import slow_mistral

llm = slow_mistral

#### Validator

class ValidatorState(MessagesState):
    completed_epics: List[Epic]
    validated_epics: Annotated[List[Epic], add]
    user_stories: List[UserStory]
    validated_user_stories: Annotated[List[FlatUserStory], add]
    amigos_max_iter: int

class ValidatorInputState(MessagesState):
    completed_epics: List[Epic]
    amigos_max_iter: int

class ValidatorOutputState(MessagesState):
    validated_epics: Annotated[List[Epic], add]

def to_pydantic(epics: List[Epic]):
    if type(epics[0]) is not Epic:
        epics = [Epic(**epic) for epic in epics]
    return epics

def product_manager_in(state: ValidatorState):
    epics = to_pydantic(state["completed_epics"])
    formated_user_stories = []
    for epic in epics:
        for i_s, story in enumerate(epic.user_stories):
            formated_user_stories.append({"user_story": story,
                                        "epic": epic.epic,
                                        "related_user_stories": [us for j_s, us in enumerate(epic.user_stories) if j_s != i_s]})
    return {"user_stories": formated_user_stories}

def product_manager_out(state: ValidatorState):
    formated_epics = {}
    for s in state["validated_user_stories"]:
        formated_epics[s.epic] = formated_epics.get(s.epic, [])
        formated_epics[s.epic].append(UserStory(**s.model_dump()))
    formated_epics = [Epic(**{"epic": k, "user_stories": v}) for k, v in formated_epics.items()]
    return {"validated_epics": formated_epics}

def task_distribution(state: ValidatorState):
    return [Send(
        "3 Amigos", 
        ThreeAmigosInputState(
            epic=s["epic"],
            user_story=s["user_story"],
            related_user_stories=s["related_user_stories"],
            max_iter=state["amigos_max_iter"]
    )) for s in state["user_stories"]]

validator_builder = StateGraph(
    ValidatorState,
    input=ValidatorInputState,
    output=ValidatorOutputState,
)

validator_builder.add_node("Product Manager Attribution", product_manager_in)
validator_builder.add_node("3 Amigos", amigos_builder.compile())
validator_builder.add_node("Product Manager Gathering", product_manager_out)

validator_builder.add_edge(START, "Product Manager Attribution")

validator_builder.add_conditional_edges(
    "Product Manager Attribution", task_distribution, ["3 Amigos"]
)
validator_builder.add_edge("3 Amigos", "Product Manager Gathering")
validator_builder.add_edge("Product Manager Gathering", END)