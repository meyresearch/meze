from meze import ColdMeze, Sofra
import sys
import json
from pathlib import Path


REPO_ROOT = Path().resolve()
EXAMPLES_DIR = REPO_ROOT / "examples"
DATA_DIR = REPO_ROOT / "data"

project_dir = DATA_DIR
project_dir = "/Users/af25016/projects/hybrid_model/"

system_name = "vim2"

hybrid_model_sofra = Sofra.from_file(
    sofra_file=f"{project_dir}/inputs/hybrid_model/protein/{system_name}/model_ezaff_sofra.json"
)
hybrid_model_sofra.build_average_charges()
