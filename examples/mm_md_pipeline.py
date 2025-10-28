from meze import ColdMeze, ColdMezeRecipe
import json

system_name = "vim2"
bridging_water = "hydroxide"
ligand_name = "ligand_11"

project_dir = f"/Users/af25016/projects/indole-carboxylates/{system_name}/"

inputs_directory = f"{project_dir}/inputs/"
# set ColdMezeRecipe including model (i.e. metal params), ligand(?)
with open(f"{inputs_directory}/model_0_recipe.json", "r") as file:
    json_recipe = json.load(file)

# load in protein files into ColdMeze
cold_meze = ColdMeze.from_files(
    recipe=ColdMezeRecipe(**json_recipe),
    pdb_file=f"{inputs_directory}/protein/vim2.fixed.pdb"
)

cold_system = cold_meze.add_ligand(f"{inputs_directory}/ligands/{ligand_name}.pdb")

# solvate <-- write solvate.py script

# heat meze

# run production