import subprocess
import sys, os, yaml

root = subprocess.run(["git", "rev-parse", "--show-toplevel"],
capture_output=True,
text=True,
encoding="utf-8").stdout.strip()

sys.path.append(root)

DEBUG = True

from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage

from typing import List, Literal, Annotated
from operator import add

from Text2Stories.utils.types import UserStory, FlatUserStory
from Text2Stories.utils.llm import slow_mistral

from copy import deepcopy

with open(os.path.join(root, "Text2Stories", "utils", "prompts.yaml"), "r") as f:
    prompts = yaml.safe_load(f)

llm = slow_mistral

class AntagonistState(MessagesState):
    epic: str
    user_story: UserStory
    related_user_stories: List[UserStory]
    related_user_stories_summary: str
    degradated_user_stories: Annotated[List[FlatUserStory], add]
    interventions: Annotated[List[List[Literal["Independent", "Negotiable", "Valuable", "Estimable", "Small", "Testable"]]], add]
    target_criteria: List[Literal["Independent", "Negotiable", "Valuable", "Estimable", "Small", "Testable"]]
    max_no_sum: int

class AntagonistInputState(MessagesState):
    epic: str
    user_story: UserStory
    related_user_stories: List[UserStory]
    target_criteria: List[Literal["Independent", "Negotiable", "Valuable", "Estimable", "Small", "Testable"]]
    max_no_sum: int

class AntagonistOutputState(MessagesState):
    degradated_user_stories: Annotated[List[FlatUserStory], add]
    interventions: Annotated[List[List[Literal["Independent", "Negotiable", "Valuable", "Estimable", "Small", "Testable"]]], add]

po = llm.with_structured_output(UserStory)

def route_input(state: AntagonistState):
    if len(state["related_user_stories"]) > state["max_no_sum"]:
        return "summarize"
    else:
        return "degrade"
    

def summarize(state: AntagonistState):
    summary = llm.invoke(
        [
            SystemMessage(prompts["Degradation"]["summarize_sys"]),
            HumanMessage(prompts["Degradation"]["summarize_human"].format(
                user_story=state["user_story"].to_str(),
                related_user_stories="\n".join([us.to_str_desc_only() for us in state["related_user_stories"]])
            ))
        ]
    )
    return {"related_user_stories_summary": summary.content}

def format_fix(state: AntagonistState):
    formated_story = po.invoke(
        [
            SystemMessage(prompts["Degradation"]["format_sys"]),
            HumanMessage(prompts["Degradation"]["format_human"].format(
                last_message=state["messages"][-1].content,
            ))
        ]
    )
    return {"degradated_user_stories": [FlatUserStory(**formated_story.model_dump(), epic=state["epic"])],
            "interventions": [state["target_criteria"]]}

def antagonist(state: AntagonistState):
    if state.get("target_criteria"):
        altered_story = llm.invoke(
            [
                SystemMessage(prompts["Degradation"]["degrade_sys"]),
                HumanMessage(prompts["Degradation"]["degrade_human"].format(
                    user_story=state["user_story"].to_str(),
                    target_criteria=", ".join(state["target_criteria"]),
                    related_user_stories="\n".join([us.to_str_desc_only() for us in state["related_user_stories"]]) \
                            if not state.get("related_user_stories_summary") \
                            else state["related_user_stories_summary"]
                ))
            ]
        )
    else:
        altered_story = deepcopy(state["user_story"])
    return {"messages": [altered_story]}

antagonist_builder = StateGraph(
    AntagonistState,
    input=AntagonistInputState,
    output=AntagonistOutputState,
)

antagonist_builder.add_node("Summarize", summarize)
antagonist_builder.add_node("Malfeasant Product Owner", antagonist)
antagonist_builder.add_node("Format", format_fix)

antagonist_builder.add_conditional_edges(
    START,
    route_input,
    {
        "summarize": "Summarize",
        "degrade": "Malfeasant Product Owner"
    }
)
antagonist_builder.add_edge("Summarize", "Malfeasant Product Owner")
antagonist_builder.add_edge("Malfeasant Product Owner", "Format")
antagonist_builder.add_edge("Format", END)