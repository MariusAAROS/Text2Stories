import subprocess
import sys, os, yaml

root = subprocess.run(["git", "rev-parse", "--show-toplevel"],
capture_output=True,
text=True,
encoding="utf-8").stdout.strip()

sys.path.append(root)

DEBUG = False

from langgraph.graph import MessagesState, StateGraph, START, END
from pydantic import ValidationError, Field
from langgraph.checkpoint.memory import MemorySaver

from typing import List, Annotated
from operator import add

from Text2Stories.utils.types import Feedback, Need, Needs
from Text2Stories.models.agents.expert import expert_builder, ExpertAgentInput
from Text2Stories.utils.llm import slow_mistral

with open(os.path.join(root, "Text2Stories", "utils", "prompts.yaml"), "r") as f:
    prompts = yaml.safe_load(f)

llm = slow_mistral

expert_memory = MemorySaver()
generator = expert_builder.compile(checkpointer=expert_memory)
evaluator = llm.with_structured_output(Feedback)

class ElicitatorState(MessagesState):
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

class ElicitatorInputState(MessagesState):
    max_iter: int
    sum_iter: int
    current_iter: int
    min_needs: int

    product: str = Field("Product to be built")
    subject: str = Field("Subject of the project")
    task: str = Field("Task to be accomplished")

class ElicitatorOutputState(MessagesState):
    needs: List[Need]
    summarized_needs: str

thread_1 = {"configurable": {"thread_id": "10"}}
thread_2 = {"configurable": {"thread_id": "11"}}

def llm_call_generator(state: ElicitatorState):
    """LLM generates a collection of needs"""

    if state.get("feedback"):
        try:
            if state.get("summarized_needs"):
                msg = generator.invoke(
                    ExpertAgentInput(messages=[
                        prompts["Elicitation"]["generate_needs_with_feedback_and_summary"].format(
                            product=state["product"],
                            tools="retriever",
                            feedback=state["feedback"],
                            summarized_needs=state["summarized_needs"],
                            new_needs=state["needs"][-state["sum_iter"]:]
                        )
                    ],
                    subject=state["subject"],
                    task=state["task"]),
                    thread_2
                )
            else:
                msg = generator.invoke(
                    ExpertAgentInput(messages=[
                        prompts["Elicitation"]["generate_needs_with_feedback"].format(
                            product=state["product"],
                            tools="retriever",
                            feedback=state["feedback"],
                            needs=state["needs"]
                        )
                    ],
                    subject=state["subject"],
                    task=state["task"]),
                    thread_2
                )
        except ValidationError as e:
            msg = Needs(needs=[])
    else:
        try:
            if state.get("summarized_needs"):
                msg = generator.invoke(
                    ExpertAgentInput(messages=[
                        prompts["Elicitation"]["generate_needs_with_summary"].format(
                            product=state["product"],
                            tools="retriever",
                            summarized_needs=state["summarized_needs"],
                            new_needs=state["needs"][-state["sum_iter"]:]
                        )
                ],
                subject=state["subject"],
                task=state["task"]),
                thread_2
            )
            else:
                msg = generator.invoke(
                    ExpertAgentInput(messages=[
                prompts["Elicitation"]["generate_needs_without_feedback_and_summary"].format(
                    product=state["product"],
                    tools="retriever",
                    needs=state["needs"]
                )
                ],
                subject=state["subject"],
                task=state["task"]),
                thread_2
            )
        except ValidationError as e:
            msg = Needs(needs=[])
    return {"needs": msg["final_response"].needs}


def llm_call_evaluator(state: ElicitatorState):
    """LLM evaluates the current collection of needs"""
    try:
        grade = evaluator.invoke(
            prompts["Elicitation"]["evaluate_needs"].format(
                needs=state["needs"][-state["sum_iter"]:],
                summarized_needs=state["summarized_needs"] if state.get("summarized_needs") else "No summary",
            )
        )
    except ValidationError as e:
        grade = Feedback(grade="unsufficient", feedback="Reformulate your Needs list please")
    return {"grade": grade.grade,
            "feedback": grade.feedback,
            "current_iter": state["current_iter"] + 1}

def summarize_needs(state: ElicitatorState):
    """Summarize the needs collected"""
    try:
        current_summary = state["summarized_needs"]
    except KeyError:
        current_summary = None
    if current_summary:
        message = prompts["Elicitation"]["update_needs_summary"].format(
            current_summary=current_summary,
            needs=state["needs"][-state["sum_iter"]:]
        )
    else:
        message = prompts["Elicitation"]["create_needs_summary"].format(
            needs=state["needs"]
        )
    
    new_summary = llm.invoke(message)
    return {"summarized_needs": new_summary.content}

# Conditional edge function to route back to joke generator or end based upon feedback from the evaluator
def route(state: ElicitatorState):
    """Route back to the generator or end based upon feedback from the evaluator"""
    keep_on = (state["current_iter"] < state["max_iter"]) and (len(state["needs"]) < state["min_needs"]) and (state["grade"] == "unsufficient")  
    if not keep_on:
        return "accepted"
    else:
        return "rejected"

# Build workflow
elicitator_builder = StateGraph(
    ElicitatorState,
    input=ElicitatorInputState,
    output=ElicitatorOutputState,
)

def summary_route(state: ElicitatorState):
    """Determine the necessity or not to summarize the newly collected needs"""
    if state["current_iter"] % state["sum_iter"] == 0 and state["current_iter"] != 0:
        return "summarize"
    else:
        return "continue"

# Add the nodes
elicitator_builder.add_node("Expert", llm_call_generator)
elicitator_builder.add_node("Product Manager", llm_call_evaluator)
elicitator_builder.add_node("Summary", summarize_needs)

# Add edges to connect nodes
elicitator_builder.add_edge(START, "Expert")
elicitator_builder.add_edge("Summary", "Product Manager")
elicitator_builder.add_conditional_edges(
    "Expert",
    summary_route,
    {
        "summarize": "Summary",
        "continue": "Product Manager",
    },
)
elicitator_builder.add_conditional_edges(
    "Product Manager",
    route,
    {  # Name returned by route_joke : Name of next node to visit
        "accepted": END,
        "rejected": "Expert",
    },
)