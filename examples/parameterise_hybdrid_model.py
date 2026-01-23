from meze import ColdMeze
import sys
import json

project_dir = sys.argv[1] 
system_name = sys.argv[2]
ligand_name = sys.argv[3]


with open(f"{project_dir}/inputs/hybrid_model/protein/{system_name}/model_ezaff_recipe.json", "r") as file:
     json_recipe = json.load(file)

parameterisation_directory = json_recipe["parameterisation_directory"]

prepared_meze = ColdMeze.load(
    filename=f"{parameterisation_directory}/{ligand_name}_meze.pkl"
)
print(prepared_meze)


prepared_meze.build_empirical_bonds()

# run once without chgfix_resids

prepared_meze.build_resp_charges(fix_ligand_charge=False) 

# then fix ligand charge, making a new directory
prepared_meze.build_resp_charges(fix_ligand_charge=True) 

# then average active site charges for fixed ligand charge
# apply to all, making a new directory again