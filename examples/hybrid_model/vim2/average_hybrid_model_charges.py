from meze import ColdMeze
import sys
import json

project_dir = "/Users/af25016/projects/meze/data/" 
project_dir = "/Users/af25016/projects/hybrid_model/"
system_name = "vim2"


# replace with reading in all ligands in the sofra!!
# ligands 

with open(f"{project_dir}/inputs/hybrid_model/protein/{system_name}/model_ezaff_sofra.json", "r") as file:
     json_recipe = json.load(file)

for i, ligand in enumerate(json_recipe.keys()):
    ligand_name = ligand
    parameterisation_directory = json_recipe[ligand_name]["parameterisation_directory"]


    prepared_meze = ColdMeze.load(
        filename=f"{parameterisation_directory}/{ligand_name}_fixed_charges_solvated.pkl"
    )


