import subprocess
import os, sys, yaml

root = subprocess.run(["git", "rev-parse", "--show-toplevel"],
capture_output=True,
text=True,
encoding="utf-8").stdout.strip()

sys.path.append(root)

DEBUG = False

#from Text2Stories.models.agents.structured_dualLM_react_expert import expert_agent
from langgraph.graph import MessagesState, StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from pydantic import Field

from typing import TypedDict

from Text2Stories.utils.types import Needs
from Text2Stories.models.tools import retriever
from Text2Stories.utils.llm import slow_mistral

with open(os.path.join(root, "Text2Stories", "utils", "prompts.yaml"), "r") as f:
    prompts = yaml.safe_load(f)

llm= slow_mistral

class ExpertAgentInput(MessagesState):
    subject: str = Field("Subject of the project")
    task: str = Field("Task to be accomplished")

class ExpertAgentOutput(TypedDict):
    final_response: Needs

class ExpertAgentState(MessagesState):
    subject: str = Field("Subject of the project")
    task: str = Field("Task to be accomplished")
    final_response: Needs

tools = [retriever]
toolmodel = llm.bind_tools(tools)
structmodel = llm.with_structured_output(Needs)

def call_model(state: ExpertAgentState):
    response = toolmodel.invoke(
        [
            SystemMessage(content=prompts["Elicitation"]["expert_sys"].format(
                                    subject=state["subject"],
                                    task=state["task"],
                                    tools=" | ".join([t.name for t in tools]))),
            HumanMessage(content="\n\n".join([s.content if isinstance(s.content, str) else "" for s in state["messages"][-3:]]))
        ]
        )
    return {"messages": [response]}

def respond(state: ExpertAgentState):
    response = structmodel.invoke(
        [
            SystemMessage(content=prompts["Elicitation"]["format_needs_sys"]),
            HumanMessage(content=state["messages"][-1].content)
        ]
    )
    return {"final_response": response}

def should_continue(state: ExpertAgentState):
    messages = state["messages"]
    last_message = messages[-1]

    if not last_message.tool_calls:
        return "respond"
    else:
        return "continue"

expert_builder = StateGraph(
    ExpertAgentState,
    input=ExpertAgentInput,
    output=ExpertAgentOutput
)

expert_builder.add_node("Agent", call_model)
expert_builder.add_node("Respond", respond)
expert_builder.add_node("Tools", ToolNode(tools))

expert_builder.set_entry_point("Agent")
expert_builder.add_conditional_edges("Agent",
                               should_continue,
                               {"respond": "Respond",
                                "continue": "Tools"})
expert_builder.add_edge("Tools", "Agent")
expert_builder.add_edge("Respond", END)