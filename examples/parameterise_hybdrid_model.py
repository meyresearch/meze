from meze import ColdMeze, ColdMezeRecipe
import sys
import json

project_dir = sys.argv[1] 
system_name = sys.argv[2]
ligand_name = sys.argv[3]


with open(f"{project_dir}/inputs/hybrid_model/protein/{system_name}/model_ezaff_recipe.json", "r") as file:
     json_recipe = json.load(file)

# parm_dir = json_recipe["parameterisation_directory"]

# cold_meze = ColdMeze.from_files(
#     recipe=ColdMezeRecipe(**json_recipe),
#     pdb_file=f"{parm_dir}/{system_name}_{ligand_name}.amber.pdb"
# )

parameterisation_directory = json_recipe["parameterisation_directory"]

prepared_meze = ColdMeze.load(
    filename=f"{parameterisation_directory}/{ligand_name}_meze.pkl"
)
print(prepared_meze)