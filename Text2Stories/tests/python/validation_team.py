import subprocess
import sys
import json

root = subprocess.run(["git", "rev-parse", "--show-toplevel"],
capture_output=True,
text=True,
encoding="utf-8").stdout.strip()

sys.path.append(root)

from langchain_mistralai import ChatMistralAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from Text2Stories.models.teams.validation import validator_builder, ValidatorInputState
from Text2Stories.utils.types import Epic

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,
    check_every_n_seconds=0.5,
    max_bucket_size=1,
)

llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0.3,
    max_retries=2,
    rate_limiter=rate_limiter,
)

with open(root+"/Data/Results/epics2.json", "r", encoding="utf-8") as f:
    epics = json.load(f)
    epics = [Epic(**e) for e in epics["epics"]]

toy_epics = [Epic(epic=e.epic, user_stories=e.user_stories[:2]) for e in epics[:2]]
toy_epics = [epic.model_dump() for epic in toy_epics]

validator = validator_builder.compile()
validator_input = ValidatorInputState(messages=[],
                                      completed_epics=toy_epics,
                                      amigos_max_iter=1)

validator_output = validator.invoke(validator_input)

report = f"""
=============================
TEST REPORT - VALIDATION TEAM
=============================

Input:
{validator_input}

Output:
{validator_output}

=============================
"""

print(report)
with open(root+"/Text2Stories/tests/reports/validation_team_report.txt", "w", encoding="utf-8") as f:
    f.write(report)