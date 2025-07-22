import subprocess
import sys, os, yaml
from copy import deepcopy

root = subprocess.run(["git", "rev-parse", "--show-toplevel"],
capture_output=True,
text=True,
encoding="utf-8").stdout.strip()

sys.path.append(root)

DEBUG = False

#from Text2Stories.models.agents.structured_dualLM_react_expert import expert_agent
from langgraph.graph import MessagesState, StateGraph, START, END
from pydantic import ValidationError
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List

from Text2Stories.utils.types import UserStory, Feedback, FlatUserStory
from Text2Stories.utils.llm import slow_mistral

with open(os.path.join(root, "Text2Stories", "utils", "prompts.yaml"), "r") as f:
    prompts = yaml.safe_load(f)

llm = slow_mistral

### Three Amigos
class ThreeAmigosState(MessagesState):
    epic: str
    user_story: UserStory
    validated_user_story: UserStory
    validated_user_stories: List[FlatUserStory]
    related_user_stories: List[UserStory]
    qa_feedback: Feedback
    dev_feedback: Feedback
    max_iter: int
    qa_current_iter: int
    dev_current_iter: int

class ThreeAmigosInputState(MessagesState):
    epic: str
    user_story: UserStory
    related_user_stories: List[UserStory]
    max_iter: int

class ThreeAmigosOutputState(MessagesState):
    epic: str
    validated_user_stories: List[FlatUserStory]

llm_po = llm.with_structured_output(UserStory)
llm_qa = llm.with_structured_output(Feedback)
llm_dev = llm.with_structured_output(Feedback)

def to_pydantic(state: ThreeAmigosState):
    """Convert the state to a pydantic model"""
    output = {}
    if type(state["user_story"]) == dict:
        output["user_story"] = UserStory(**state["user_story"], epic=state["epic"])
        output["validated_user_story"] = UserStory(**state["user_story"], epic=state["epic"])
        output["validated_user_stories"] = [FlatUserStory(**state["user_story"], epic=state["epic"])]
    else:
        output["user_story"] = deepcopy(state["user_story"])
        output["validated_user_story"] = deepcopy(state["user_story"])
        output["validated_user_stories"] = [deepcopy(state["user_story"])]
    if len(state["related_user_stories"]) > 0 and type(state["related_user_stories"][0]) == dict:
        casted_related_user_stories = [UserStory(**us) for us in state["related_user_stories"]]
        output["related_user_stories"] = casted_related_user_stories
    else:
        output["related_user_stories"] = deepcopy(state["related_user_stories"])
    output["qa_current_iter"] = 0
    output["dev_current_iter"] = 0
    output["qa_feedback"] = Feedback(grade="unsufficient", feedback="No feedback yet")
    output["dev_feedback"] = Feedback(grade="unsufficient", feedback="No feedback yet")
    return output
    

def product_owner(state: ThreeAmigosState):
    try:
        us = llm_po.invoke(
            [
                SystemMessage(prompts["Validation"]["po_sys"]),
                HumanMessage(prompts["Validation"]["po_human"].format(
                    user_story=state["user_story"].to_str_no_invest(),
                    validated_user_story=state["validated_user_story"].to_str_no_invest(),
                    qa_feedback=state["qa_feedback"].feedback,
                    dev_feedback=state["dev_feedback"].feedback
                ))
            ]
        )
    except ValidationError as e:
        us = state["validated_user_story"] if not state["validated_user_story"] == state["user_story"] else state["user_story"]
    
    #INCLUDE FALLBACK VALUES FOR INVEST CRITERIA
    for field in ["independent", "negotiable", "valuable", "estimable", "small", "testable"]:
        if getattr(us, field) is None:
            setattr(us, field, getattr(state["validated_user_story"], field) if getattr(state["validated_user_story"], field) is not None else False)

    return {"validated_user_story": us,
            "validated_user_stories": [FlatUserStory(**us.model_dump(), epic=state["epic"])]}

def quality_assurance_tester(state: ThreeAmigosState):
    try:
        feedback = llm_qa.invoke(
            [
                SystemMessage(prompts["Validation"]["qa_sys"]),
                HumanMessage(prompts["Validation"]["qa_human"].format(
                    validated_user_story=state["validated_user_story"].to_str_no_invest(),
                    related_user_stories="\n".join([r.to_str_no_invest() for r in state["related_user_stories"]]),
                ))
            ]
        )
    except ValidationError as e:
        feedback = Feedback(grade="unsufficient", feedback="Reformulate your User Story please")

    return {"qa_feedback": feedback,
            "qa_current_iter": state["qa_current_iter"] + 1}

def developer(state: ThreeAmigosState):
    try:
        feedback = llm_dev.invoke(
            [
                SystemMessage(prompts["Validation"]["dev_sys"]),
                HumanMessage(prompts["Validation"]["dev_human"].format(
                    validated_user_story=state["validated_user_story"].to_str_no_invest(),
                    related_user_stories="\n".join([r.to_str_no_invest() for r in state["related_user_stories"]])
                ))
            ]
        )
    except ValidationError as e:
        feedback = Feedback(grade="unsufficient", feedback="Reformulate your User Story please")

    return {"dev_feedback": feedback,
            "dev_current_iter": state["dev_current_iter"] + 1}

def three_amigos(state: ThreeAmigosState):
    auth_qa = (state["qa_current_iter"] < state["max_iter"]) and (state["qa_feedback"].grade == "unsufficient")
    auth_dev = (state["dev_current_iter"] < state["max_iter"]) and (state["dev_feedback"].grade == "unsufficient")

    if auth_dev:
        return "dev"
    elif auth_qa:
        return "qa"
    elif not auth_dev and not auth_qa:
        return "end"
    else:
        raise ValueError("Invalid feedback grades")

amigos_builder = StateGraph(
    ThreeAmigosState,
    input=ThreeAmigosInputState,
    output=ThreeAmigosOutputState,
)

amigos_builder.add_node("Product Owner", product_owner)
amigos_builder.add_node("QA Tester", quality_assurance_tester)
amigos_builder.add_node("Developer", developer)
amigos_builder.add_node("State Check", to_pydantic)

amigos_builder.add_edge(START, "State Check")
amigos_builder.add_edge("State Check", "Product Owner")
amigos_builder.add_conditional_edges("Product Owner",
                                        three_amigos,
                                        {
                                            "qa": "QA Tester",
                                            "dev": "Developer",
                                            "end": END
                                        }
                                        )
amigos_builder.add_edge("QA Tester", "Product Owner")
amigos_builder.add_edge("Developer", "Product Owner")