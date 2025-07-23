import subprocess
import os, sys, yaml
import json
from tqdm import tqdm
from collections import defaultdict

root = subprocess.run(["git", "rev-parse", "--show-toplevel"],
capture_output=True,
text=True,
encoding="utf-8").stdout.strip()

sys.path.append(root)
save_dir = os.path.join(root, "Experiments", "Ablation", "output")
save_path = os.path.join(save_dir, "llm+CoT.json")

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langgraph.checkpoint.memory import InMemorySaver
from langchain_mistralai import ChatMistralAI
from Text2Stories.utils.types import Epics, Need, Epic, Needs, UserStory, FlatUserStory
from pydantic import BaseModel, ValidationError
from typing import Annotated, List, TypedDict
from operator import add

from langgraph.graph import StateGraph, START, END, MessagesState

TOTAL_USER_STORIES = 100
FROM_EXISTING = False

with open(os.path.join(root, "Experiments", "Ablation", "ablation_prompts.yaml"), "r") as f:
    prompts = yaml.safe_load(f)

slow_rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,
    check_every_n_seconds=0.5,
    max_bucket_size=1,
)

llm = ChatMistralAI(
    model_name="mistral-large-latest",
    temperature=0.3,
    max_retries=2,
    rate_limiter=slow_rate_limiter
)

llm4needs = llm.with_structured_output(Needs)
llm4epics = llm.with_structured_output(Epics)
llm4userstory = llm.with_structured_output(UserStory)

def merge_dicts(dict1, dict2):
    merged = defaultdict(list)
    for d in [dict1, dict2]:
        for key, value in d.items():
            merged[key].extend(value)
    return dict(merged)

def quick_save(data: BaseModel, erase=True):
    if erase:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data.model_dump(), f, indent=4)
    else:
        try:
            with open(save_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            data = merge_dicts(existing_data, data.model_dump())
            data = Epics(**data)
        except:
            pass
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data.model_dump(), f, indent=4)

def to_pydantic(epics: List[Epic]):
    if type(epics[0]) is not Epic:
        epics = [Epic(**epic) for epic in epics]
    return epics

def epics2dict(epics: List[Epic]) -> List[dict]:
    epics = to_pydantic(epics)
    formated_user_stories = []
    for epic in epics:
        for i_s, story in enumerate(epic.user_stories):
            formated_user_stories.append({"user_story": story,
                                        "epic": epic.epic,
                                        "related_user_stories": [us for j_s, us in enumerate(epic.user_stories) if j_s != i_s]})
    return formated_user_stories

def dict2epics(dict: List[dict]) -> List[Epic]:
    formated_epics = {}
    if dict != []:
        for s in dict:
            formated_epics[s.epic] = formated_epics.get(s.epic, [])
            formated_epics[s.epic].append(UserStory(**s.model_dump()))
        formated_epics = [Epic(**{"epic": k, "user_stories": v}) for k, v in formated_epics.items()]
    return formated_epics

class State(MessagesState):
    product: str
    topic: str
    needs: Annotated[List[Need], add]
    epics: Annotated[List[Epics], add]
    validated_epics: Annotated[List[Epics], add]

class OutputState(TypedDict):
    validated_epics: Annotated[List[Epics], add]

builder = StateGraph(
    State,
    output=OutputState
)

def elicitation(state: State) -> State:
    try:
        output = llm4needs.invoke(
            [
                SystemMessage(prompts["CoT"]["elicitation"]["sys"].format(
                    product=state["product"],
                    topic=state["topic"]
                )),
                HumanMessage(prompts["CoT"]["elicitation"]["user"].format(
                    product=state["product"],
                    needs="\n".join([need.need for need in state["needs"]]) if state["needs"] else "No needs yet.",
                ))
            ]
        )
    except ValidationError as e:
        print("Error in elicitation:", e)
        output = Needs(needs=[])
    return {"needs": output.needs}

def scheduling(state: State) -> State:
    try:
        output = llm4epics.invoke(
            [
                SystemMessage(prompts["CoT"]["scheduling"]["sys"]),
                HumanMessage(prompts["CoT"]["scheduling"]["user"].format(
                    product=state["product"],
                    needs="\n".join([need.need for need in state["needs"]]),
                    user_stories="\n".join([us.user_story for epic in state["epics"] for us in epic.user_stories]) if state["epics"] else "No user stories yet.",
                ))
            ]
        )
    except ValidationError as e:
        print("Error in scheduling:", e)
        output = Epics(epics=[])
    return {"epics": output.epics}

def verification(state: State) -> State:
    user_stories = epics2dict(state["epics"])
    output = []
    for us in user_stories:
        try:
            current = llm4userstory.invoke(
                [
                    SystemMessage(prompts["CoT"]["verification"]["sys"]),
                    HumanMessage(prompts["CoT"]["verification"]["user"].format(
                        user_story=us["user_story"].to_str_no_invest(),
                        related_user_stories="\n".join([r_us.user_story for r_us in us["related_user_stories"]]) if us["related_user_stories"] else "No related user stories.",
                    ))
                ]
            )
            output.append(FlatUserStory(**current.model_dump(), epic=us["epic"]))
        except ValidationError as e:
            print("Error in verification:", e)
    output = dict2epics(output)
    return {"validated_epics": output}

builder.add_node("Elicitation", elicitation)
builder.add_node("Scheduling", scheduling)
builder.add_node("Verification", verification)

builder.add_edge(START, "Elicitation")
builder.add_edge("Elicitation", "Scheduling")
builder.add_edge("Scheduling", "Verification")
builder.add_edge("Verification", END)

graph = builder.compile()

if FROM_EXISTING:
    with open(save_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        output = Epics(**data)
    
else:
    output = Epics(epics=[])
    if os.path.exists(save_path):
         os.remove(save_path)

sum_stories = lambda epics: sum([len(epic.user_stories) if epic.user_stories else 0 for epic in epics.epics]) if epics.epics is not [] else 0

pbar = tqdm(total=TOTAL_USER_STORIES, desc="Generating User Stories")
while sum_stories(output) < TOTAL_USER_STORIES:
    current = graph.invoke(
        State(product="loan granting application for individuals",
               topic="loan granting processes")
    )
    typed_current = Epics(epics=current["validated_epics"])
    output = merge_dicts(output.model_dump(), typed_current.model_dump())
    output = Epics(**output)
    pbar.n = sum_stories(output)
    pbar.refresh()
    quick_save(output, erase=False)

quick_save(output, erase=True)
print("Output saved to", save_path)
print("Output:\n", json.dumps(output.model_dump(), indent=4))