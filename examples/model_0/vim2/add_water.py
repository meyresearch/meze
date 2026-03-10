from meze import ColdMeze, ColdMezeRecipe
import json
import os
import meze
from pathlib import Path

REPO_ROOT = Path().resolve()  
EXAMPLES_DIR = REPO_ROOT / "examples"
DATA_DIR = REPO_ROOT / "data"

system_name = "vim2"
bridging_water = "hydroxide"
ligand_name = "ligand_11"

project_dir = DATA_DIR

# set ColdMezeRecipe including model (i.e. metal params), ligand(?)
with open(f"{project_dir}/inputs/model_0/protein/{system_name}/model_0_recipe.json", "r") as file:
    json_recipe = json.load(file)

json_recipe["path_to_engine"] = os.path.join(
        os.environ["AMBERHOME"], "bin", "pmemd.cuda"
    )
# load in protein files into ColdMeze
cold_meze = ColdMeze.from_files(
    recipe=ColdMezeRecipe(**json_recipe),
    pdb_file=f"{project_dir}/inputs/model_0/protein/{system_name}/vim2.fixed.pdb",
)

cold_meze_with_lig = cold_meze.add_ligand(
    ligand_file=f"{project_dir}/inputs/model_0/ligands/{system_name}/{ligand_name}.pdb",
    ligand_charge=-1
)

#TODO make non standard res a union of Ligand and List[Ligand]
cold_system = cold_meze_with_lig.add_non_standard_residue(
    file=f"{project_dir}/inputs/model_0/protein/{system_name}/MOH.pdb",
    charge=-1,
    atom_type="amber"
)

print(cold_system)

#TODO: put solvation options into MezeRecipe
solvate_dir = f"{project_dir}/inputs/model_0/protein/{system_name}/solvate_{ligand_name}_bound/"


solvated_meze = cold_system.add_water(directory=solvate_dir)

print(solvated_meze)