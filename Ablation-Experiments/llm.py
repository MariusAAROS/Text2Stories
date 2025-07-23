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
save_path = os.path.join(save_dir, "llm.json")

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langgraph.checkpoint.memory import InMemorySaver
from langchain_mistralai import ChatMistralAI
from Text2Stories.utils.types import Epics
from pydantic import BaseModel

from langgraph.graph import Graph, START, END

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
    model="mistral-large-latest",
    temperature=0.3,
    max_retries=2,
    rate_limiter=slow_rate_limiter
).with_structured_output(Epics)

def call_llm(prompt) -> Epics:
    output = llm.invoke(prompt)
    return output

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

builder = Graph()

builder.add_node("call", call_llm)
builder.add_edge(START, "call")
builder.add_edge("call", END)

memory = InMemorySaver()

config = {"configurable": {"thread_id": "1"}}
graph = builder.compile(checkpointer=memory)

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
        [
            SystemMessage(prompts["basic"]["sys"]),
            HumanMessage(prompts["basic"]["user"].format(
                product="loan granting application for individuals",
                user_stories="\n".join([us.user_story for epic in output.epics for us in epic.user_stories])))
        ],
        config=config,
    )
    output = merge_dicts(output.model_dump(), current.model_dump())
    output = Epics(**output)
    pbar.n = sum_stories(output)
    pbar.refresh()
    quick_save(output, erase=False)

quick_save(output, erase=True)
print("Output saved to", save_path)
print("Output:\n", json.dumps(output.model_dump(), indent=4))