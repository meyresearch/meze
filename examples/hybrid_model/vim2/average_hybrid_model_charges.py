from meze import ColdMeze
import sys
import json

project_dir = "/Users/af25016/projects/meze/data/" 
system_name = "vim2"
ligand_name = "ligand_11"
# ligands 

with open(f"{project_dir}/inputs/hybrid_model/protein/{system_name}/model_ezaff_sofra.json", "r") as file:
     json_recipe = json.load(file)

parameterisation_directory = json_recipe["ligand_11"]["parameterisation_directory"]


prepared_meze = ColdMeze.load(
    filename=f"{parameterisation_directory}/{ligand_name}_fixed_charges_solvated.pkl"
)


