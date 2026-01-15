from meze import ColdMeze, ColdMezeRecipe
import json
import os

workdir = os.getcwd()
print(workdir)

system_name = "l1"
ligand_name = "lig53"

with open(f"{workdir}/data/inputs/model_0/protein/{system_name}/model_0_recipe.json", "r") as file:
    json_recipe = json.load(file)

cold_meze = ColdMeze.from_files(
    recipe=ColdMezeRecipe(**json_recipe),
    pdb_file=f"{workdir}/data/inputs/model_0/protein/{system_name}/l1_no_elements.pdb"
)
