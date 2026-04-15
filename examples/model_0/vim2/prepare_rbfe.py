from meze import ColdMeze, ColdMezeRecipe, Sofra
import os
import json
import sys
from pathlib import Path
import glob

REPO_ROOT = Path().resolve()
EXAMPLES_DIR = REPO_ROOT / "examples"
DATA_DIR = REPO_ROOT / "data"

system_name = "vim2"

project_dir = DATA_DIR

#DEBUGGING
project_dir = "/Users/af25016/projects/vim2-model-0/"

model_0_sofra = Sofra.from_file(
    sofra_file=f"{project_dir}/inputs/model_0/protein/{system_name}/model_0_sofra.json",
    directory=project_dir
)

with open(f"{project_dir}/inputs/model_0/protein/{system_name}/model_0_recipe.json", "r") as file:
    json_recipe = json.load(file)

json_recipe["path_to_engine"] = os.path.join(
        os.environ["AMBERHOME"], "bin", "sander"
)

ligand_path = f"{project_dir}/inputs/model_0/ligands/{system_name}/"

ligand_files = sorted(glob.glob(f"{ligand_path}/ligand_*/ligand_*.pdb"))

model_0_sofra.set_ligand_network(
    pdb_files=ligand_files,
    directory=f"{project_dir}/inputs/model_0/protein/{system_name}/"
)
