import subprocess
import os, sys
import json
from tqdm import tqdm
from collections import defaultdict
from pydantic import BaseModel

root = subprocess.run(["git", "rev-parse", "--show-toplevel"],
capture_output=True,
text=True,
encoding="utf-8").stdout.strip()

sys.path.append(root)
save_dir = os.path.join(root, "Experiments", "Ablation", "output")
load_path = os.path.join(save_dir, "t2s_gen.json")
save_path = os.path.join(save_dir, "t2s_deg.json")


from Text2Stories.utils.types import Epics
from Text2Stories.models.teams.degradation import degradator_builder, DegradatorInputState

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

with open(load_path, "r", encoding="utf-8") as f:
    data = json.load(f)
    gen = Epics(epics=data["epics"])

graph = degradator_builder.compile()

raw = graph.invoke(DegradatorInputState(
    epics=gen.epics,
    max_no_sum=5,
    target_criteria=["independent", "negotiable", "valuable", "estimable", "small", "testable"],
    mode="rand",
    ))
output = Epics(epics=raw["degradated_epics"])
quick_save(output, erase=True)
print("Output saved to", save_path)
print("Output:\n", json.dumps(output.model_dump(), indent=4))