from meze import HotMeze, HotMezeRecipe
import os
import sys
import json

project_dir = sys.argv[1] 
ligand_name = sys.argv[2]
repeat = sys.argv[3]
system_name = "vim2"

# set ColdMezeRecipe including model (i.e. metal params), ligand(?)

with open(f"{project_dir}/inputs/hybrid_model/protein/{system_name}/hybrid_model_recipe.json", "r") as file:
    json_recipe = json.load(file)

json_recipe["path_to_engine"] = os.path.join(
    os.environ["AMBERHOME"], "bin", "pmemd.cuda"        
)

#TODO make a separate coldmezerecipe.json and a hotmezerecipe.json
json_recipe["runtime"] = 150
json_recipe["dt"] = 0.002

equil_dir = f"{project_dir}/equilibration/hybrid_model/bound/{ligand_name}/repeat_{repeat}/"

hot_meze = HotMeze.from_files(
    recipe=HotMezeRecipe(**json_recipe),
    topology=f"{equil_dir}/08_free/next.prm7",
    coordinates=f"{equil_dir}/08_free/next.rst7",
    restraint_file=f"{equil_dir}/restraints.RST"
) 

production_dir = os.path.join(project_dir, "outputs", "hybrid_model", f"{ligand_name}", f"repeat_{repeat}")
os.makedirs(production_dir, exist_ok=True)

hot_meze.run(
    process_name="prod",
    workdir=production_dir
)
