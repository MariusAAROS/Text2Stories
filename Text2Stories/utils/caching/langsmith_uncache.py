import os
from langsmith import utils as langsmith_utils
from dotenv import load_dotenv

import subprocess
import sys

root = subprocess.run(["git", "rev-parse", "--show-toplevel"],
capture_output=True,
text=True,
encoding="utf-8").stdout.strip()

sys.path.append(root)

print("Loaded Variables \n")

print(langsmith_utils.get_env_var("PROJECT"))
print(langsmith_utils.get_env_var("TRACING"))
print(langsmith_utils.get_env_var("ENDPOINT"))
print(langsmith_utils.get_env_var("API_KEY"))

langsmith_utils.get_env_var.cache_clear()

print("\nCleared Cache\n")

load_dotenv(os.path.join(root, ".env"), override=True)

print(langsmith_utils.get_env_var("PROJECT"))
print(langsmith_utils.get_env_var("TRACING"))
print(langsmith_utils.get_env_var("ENDPOINT"))
print(langsmith_utils.get_env_var("API_KEY"))

print("\nVariables Reloaded")