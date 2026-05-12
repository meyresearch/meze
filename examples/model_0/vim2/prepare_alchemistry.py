from meze import ColdMeze, ColdMezeRecipe, Sofra
import os
import json
import sys
from pathlib import Path
import glob


# UNCOMMENT BEFORE COMMIT
# project_dir = sys.argv[1]
# system_name = sys.argv[2]
# ligand_1 = sys.argv[3]
# ligand_2 = sys.argv[4]

#DEBUGGING
project_dir = "/Volumes/external_harddrive/projects/vim2-model-0/"
system_name = "vim2"
ligand_1 = "ligand_1"
ligand_2 = "ligand_4"

model_0_sofra = Sofra.from_file(
    sofra_file=f"{project_dir}/inputs/model_0/protein/{system_name}/model_0_sofra.json",
    directory=project_dir
)

with open(f"{project_dir}/inputs/model_0/protein/{system_name}/model_0_recipe.json", "r") as file:
    json_recipe = json.load(file)

json_recipe["path_to_engine"] = os.path.join(
        os.environ["AMBERHOME"], "bin", "sander"
)

# read in equilibrated bound and unbound systems

# merge in both 

# prep fep windows

# write submission scripts
