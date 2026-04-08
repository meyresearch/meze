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

try: #DEBUGGING; for normal running, remove try/except block
     model_0_sofra = Sofra.from_file(f"{project_dir}/inputs/model_0/protein/{system_name}/model_0_sofra.json")
except RuntimeError as e:
     print(e)
     print("Constructing Sofra manually")

with open(f"{project_dir}/inputs/model_0/protein/{system_name}/model_0_sofra.json", "r") as file:
     sofra_file = json.load(file)

with open(f"{project_dir}/inputs/model_0/protein/{system_name}/model_0_recipe.json", "r") as file:
    json_recipe = json.load(file)

json_recipe["path_to_engine"] = os.path.join(
        os.environ["AMBERHOME"], "bin", "pmemd.cuda"
    )

ligand_path = f"{project_dir}/inputs/model_0/ligands/{system_name}/"

mezes = []
for i, ligand_name in enumerate(sofra_file.keys()):
     input_dir = f"{project_dir}/inputs/model_0/protein/{system_name}/solvate_{ligand_name}_bound"

     solvated_meze = ColdMeze.from_files(
          topology=f"{input_dir}/{ligand_name}_complex_solv.prmtop",
          coordinates=f"{input_dir}/{ligand_name}_complex_solv.inpcrd",
          recipe=ColdMezeRecipe(**json_recipe)
     )
     mezes.append(solvated_meze)
     pickle_file = solvated_meze.save(
          filename=f"{input_dir}/{ligand_name}_solvated.pkl"
     )

     solvated_meze.add_to_sofra(
     f"{project_dir}/inputs/model_0/protein/{system_name}/model_0_sofra.json",
     key=ligand_name,
     pickle_file=pickle_file
     )
try: #DEBUGGING; for normal running, remove try/except block
     model_0_sofra = Sofra.from_file(f"{project_dir}/inputs/model_0/protein/{system_name}/model_0_sofra.json")
except RuntimeError as e:
     print(e)
     print("Constructing Sofra manually")


    
ligand_files = sorted(glob.glob(f"{ligand_path}/ligand_*/ligand_*.pdb"))

for i, ligand_file in enumerate(ligand_files):
    print(i, ligand_file)
    
         