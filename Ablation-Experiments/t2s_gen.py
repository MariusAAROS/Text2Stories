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
save_path = os.path.join(save_dir, "t2s_gen.json")


from Text2Stories.utils.types import Epics
from Text2Stories.models.flows.sequential_generation import us_generator_builder, USGeneratorInputState

TOTAL_USER_STORIES = 100
FROM_EXISTING = False

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


graph = us_generator_builder.compile()

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
if FROM_EXISTING:
    pbar.n = sum_stories(output)
    pbar.refresh()
while sum_stories(output) < TOTAL_USER_STORIES:
    current = graph.invoke(
        USGeneratorInputState(product="loan granting application for individuals",
               subject="loan granting processes",
               task="Creating a loan granting application for individuals",
               amigos_max_iter=2,
               sum_iter=15,
               min_needs=30,
               max_iter=1)
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