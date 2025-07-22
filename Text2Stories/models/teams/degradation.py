import subprocess
import sys
import random

root = subprocess.run(["git", "rev-parse", "--show-toplevel"],
capture_output=True,
text=True,
encoding="utf-8").stdout.strip()

sys.path.append(root)

DEBUG = True

from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.constants import Send

from typing import List, Literal, TypedDict, Annotated
from operator import add

from Text2Stories.utils.types import UserStory, Epic, FlatUserStory
from Text2Stories.models.teams.antagonist import AntagonistInputState, antagonist_builder
from Text2Stories.utils.llm import slow_mistral

llm = slow_mistral

class DegradatorState(MessagesState):
    epics: List[Epic]
    degradated_epics: List[Epic]
    user_stories: List[FlatUserStory]
    degradated_user_stories: Annotated[List[FlatUserStory], add]
    max_no_sum: int
    interventions: Annotated[List[List[Literal["Independent", "Negotiable", "Valuable", "Estimable", "Small", "Testable"]]], add]
    formated_interventions: List
    target_criteria: List[Literal["Independent", "Negotiable", "Valuable", "Estimable", "Small", "Testable"]]
    mode: Literal["rand", "spec"]

class DegradatorInputState(MessagesState):
    epics: List[Epic]
    max_no_sum: int
    target_criteria: List[Literal["Independent", "Negotiable", "Valuable", "Estimable", "Small", "Testable"]]
    mode: Literal["rand", "spec"]
    
class DegradatorOutputState(TypedDict):
    degradated_epics: List[Epic]
    formated_interventions: List

def to_pydantic(epics: List[Epic]):
    if type(epics[0]) is not Epic:
        epics = [Epic(**epic) for epic in epics]
    return epics

def product_manager_in(state: DegradatorState):
    epics = to_pydantic(state["epics"])
    formated_user_stories = []
    for epic in epics:
        for i_s, story in enumerate(epic.user_stories):
            formated_user_stories.append({"user_story": story,
                                        "epic": epic.epic,
                                        "related_user_stories": [us for j_s, us in enumerate(epic.user_stories) if j_s != i_s]})
    return {"user_stories": formated_user_stories}

def product_manager_out(state: DegradatorState):
    formated_epics = {}
    for s in state["degradated_user_stories"]:
        formated_epics[s.epic] = formated_epics.get(s.epic, [])
        formated_epics[s.epic].append(UserStory(**s.model_dump()))
    formated_epics = [Epic(**{"epic": k, "user_stories": v}) for k, v in formated_epics.items()]

    formated_interventions = []
    i = 0
    for e in formated_epics:
        temp = []
        for _ in e.user_stories:
            temp.append(state["interventions"][i])
            i += 1
        formated_interventions.append(temp)

    return {"degradated_epics": formated_epics,
            "formated_interventions": formated_interventions}

def task_distribution(state: DegradatorState):
    cardinality = len(state["user_stories"])
    if state["mode"] == "rand":
        target_criterias = [
            random.sample(
                ["independent", "negotiable", "valuable", "estimable", "small", "testable"], 
                random.randint(1, 6)) for _ in range(cardinality)
            ]
    elif state["mode"] == "spec":
        target_criterias = [state["target_criteria"] for _ in range(cardinality)]
    else:
        raise ValueError("The mode must be either rand or spec")
    
    #consistency check between criteria and user stories - you should not degrade a critiria that is already degraded
    for us, cs in zip(state["user_stories"], target_criterias):
        for c in cs:
            if not getattr(us["user_story"], c):
                cs.remove(c)
    return [Send(
        "Antagonist",
        AntagonistInputState(
            epic=s["epic"],
            user_story=s["user_story"],
            related_user_stories=s["related_user_stories"],
            target_criteria=c,
            max_no_sum=state["max_no_sum"]
        )
    ) for s, c in zip(state["user_stories"], target_criterias)]

degradator_builder = StateGraph(
    DegradatorState,
    input=DegradatorInputState,
    output=DegradatorOutputState,
)

degradator_builder.add_node("Product Manager Attribution", product_manager_in)
degradator_builder.add_node("Antagonist", antagonist_builder.compile())
degradator_builder.add_node("Product Manager Gathering", product_manager_out)

degradator_builder.add_edge(START, "Product Manager Attribution")
degradator_builder.add_conditional_edges("Product Manager Attribution",
                                         task_distribution,
                                         ["Antagonist"])
degradator_builder.add_edge("Antagonist", "Product Manager Gathering")
degradator_builder.add_edge("Product Manager Gathering", END)