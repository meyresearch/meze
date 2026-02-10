from meze import ColdMeze
import sys
import json

project_dir = sys.argv[1] 
system_name = sys.argv[2]

# ligands 

with open(f"{project_dir}/inputs/hybrid_model/protein/{system_name}/model_ezaff_recipe.json", "r") as file:
     json_recipe = json.load(file)

parameterisation_directory = json_recipe["parameterisation_directory"]


prepared_meze = ColdMeze.load(
    filename=f"{parameterisation_directory}/{ligand_name}_fixed_charges_solvated.pkl"
)


