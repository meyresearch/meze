from meze import Ligand
import json
import os
import meze
from pathlib import Path

REPO_ROOT = Path().resolve()  
EXAMPLES_DIR = REPO_ROOT / "examples"
DATA_DIR = REPO_ROOT / "data"

system_name = "vim2"
ligand_name = "ligand_11"

project_dir = DATA_DIR

# set ColdMezeRecipe including model (i.e. metal params), ligand(?)
with open(f"{project_dir}/inputs/model_0/protein/{system_name}/model_0_recipe.json", "r") as file:
    json_recipe = json.load(file)


# load in protein files into ColdMeze

cold_ligand = Ligand(
    file=f"{project_dir}/inputs/model_0/protein/{system_name}/solvate_{ligand_name}_bound/{ligand_name}.mol2",
    charge=-1,
    parameterised=True,
    frcmod_file=f"{project_dir}/inputs/model_0/protein/{system_name}/solvate_{ligand_name}_bound/{ligand_name}.frcmod"
)


#TODO: put solvation options into MezeRecipe
solvate_dir = f"{project_dir}/inputs/model_0/ligands/{system_name}/solvate_{ligand_name}_unbound/"


solvated_meze = cold_ligand.add_water(directory=solvate_dir)

