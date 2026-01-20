from meze import ColdMeze, ColdMezeRecipe
import sys
import json

project_dir = sys.argv[1] 
system_name = sys.argv[2]
ligand_name = sys.argv[3]


# set ColdMezeRecipe including model (i.e. metal params), ligand(?)
with open(f"{project_dir}/inputs/hybrid_model/protein/{system_name}/model_ezaff_recipe.json", "r") as file:
    json_recipe = json.load(file)

parm_dir = f"{project_dir}/outputs/{system_name}/{ligand_name}/01_mcpb_parameterisation/"

cold_meze = ColdMeze.from_files(
    recipe=ColdMezeRecipe(**json_recipe),
    pdb_file=f"{parm_dir}/{system_name}.amber.pdb",
    parameterisation_directory=parm_dir
)

