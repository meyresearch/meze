from meze import ColdMeze, ColdMezeRecipe
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

with open(f"{project_dir}/inputs/model_0/protein/{system_name}/model_0_sofra.json", "r") as file:
     json_recipe = json.load(file)

ligand_path = f"{project_dir}/inputs/model_0/ligands/{system_name}/"

# for i, ligand_name in enumerate(json_recipe.keys()):
    
ligand_files = sorted(glob.glob(f"{ligand_path}/ligand_*/ligand_*.pdb"))

for i, ligand_file in enumerate(ligand_files):
    print(i, ligand_file)
    
         