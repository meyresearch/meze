from meze import ColdMeze
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

for i, ligand in enumerate(json_recipe.keys()):
    ligand_name = ligand
    parameterisation_directory = json_recipe[ligand_name]["parameterisation_directory"]


    prepared_meze = ColdMeze.load(
        filename=f"{parameterisation_directory}/{ligand_name}_fixed_charges_solvated.pkl"
    )

    print(prepared_meze)
