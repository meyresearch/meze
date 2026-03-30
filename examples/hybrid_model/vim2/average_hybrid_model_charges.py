from meze import ColdMeze, build_average_charges
import sys
import json
from pathlib import Path


REPO_ROOT = Path().resolve()
EXAMPLES_DIR = REPO_ROOT / "examples"
DATA_DIR = REPO_ROOT / "data"

project_dir = DATA_DIR

system_name = "vim2"

with open(f"{project_dir}/inputs/hybrid_model/protein/{system_name}/model_ezaff_sofra.json", "r") as file:
     json_recipe = json.load(file)

prepared_meze_list = []
for i, ligand in enumerate(json_recipe.keys()):
    ligand_name = ligand
    parameterisation_directory = json_recipe[ligand_name]["parameterisation_directory"]

    prepared_meze_list.append(ColdMeze.load(
        filename=f"{parameterisation_directory}/{ligand_name}_fixed_charges_solvated.pkl"
    ))

new_meze_list = build_average_charges(prepared_meze_list)