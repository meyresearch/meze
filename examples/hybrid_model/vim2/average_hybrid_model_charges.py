from meze import ColdMeze, build_average_charges
import sys
import json
from pathlib import Path


REPO_ROOT = Path().resolve()
EXAMPLES_DIR = REPO_ROOT / "examples"
DATA_DIR = REPO_ROOT / "data"

project_dir = DATA_DIR

system_name = "vim2"

with open(f"{project_dir}/inputs/hybrid_model/protein/{system_name}/model_ezaff_sofra.json", "r") as file:
     json_recipe = json.load(file)

prepared_meze_list = []
ligand_names = []
for i, ligand in enumerate(json_recipe.keys()):
    ligand_names.append(ligand)
    parameterisation_directory = json_recipe[ligand]["parameterisation_directory"]

    prepared_meze_list.append(ColdMeze.load(
        filename=f"{parameterisation_directory}/{ligand}_fixed_charges_solvated.pkl"
    ))

new_meze_list = build_average_charges(prepared_meze_list, ligand_names=ligand_names)

for i, meze in enumerate(new_meze_list):
    meze.save(
        filename=f"{meze.parameterisation_directory}/{ligand_names[i]}_avg_charges"
    )

    solvated = meze.add_water(
        directory=meze.parameterisation_directory,
        mcpbpy_tleap_file=meze.tleap_input_file
    )

    pickled_file = solvated.save(
        filename=f"{meze.parameterisation_directory}/{ligand_names[i]}_avg_charges_solv"
    )

    solvated.add_to_sofra(
        key=ligand_names[i],
        filename=f"{project_dir}/inputs/hybrid_model/protein/{system_name}/model_ezaff_sofra.json",
        pickle_file=pickled_file
    )