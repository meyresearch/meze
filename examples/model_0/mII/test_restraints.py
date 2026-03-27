
from meze import ColdMeze, ColdMezeRecipe
import os
import json
import sys

project_dir = "/Users/af25016/projects/meze/data/"
system_name = "mII"
ligand_name = "m5g0"

# set ColdMezeRecipe including model (i.e. metal params), ligand(?)
with open(f"{project_dir}/inputs/model_0/protein/{system_name}/model_0_recipe.json", "r") as file:
    json_recipe = json.load(file)

json_recipe["path_to_engine"] = os.path.join(
        os.environ["AMBERHOME"], "bin", "sander"
    )

input_dir = f"{project_dir}/inputs/model_0/protein/{system_name}/"

solvated_meze = ColdMeze.from_files(
    topology=f"{input_dir}/complex.prm7",  # parameterise + solvate separately in tleap
    coordinates=f"{input_dir}/complex.rst7", # parameterise + solvate separately in tleap
    recipe=ColdMezeRecipe(**json_recipe),
    non_standard_residues= {
        "0YB": {"charge": 0, "atom_type": "glycam"}, 
        "ROH": {"charge": 0, "atom_type": "glycam"}, 
        "4YA": {"charge": 0, "atom_type": "glycam"}, 
        "0MA": {"charge": 0, "atom_type": "glycam"}, 
        "2MA": {"charge": 0, "atom_type": "glycam"}, 
        "VMA": {"charge": 0, "atom_type": "glycam"}, 
        "VMB": {"charge": 0, "atom_type": "glycam"}
    },
)

restraints = solvated_meze.build_distance_restraints()
solvated_meze.write_restrained_atoms_pdb(f"{project_dir}/inputs/model_0/protein/{system_name}/restrained_atoms.pdb", restraints=restraints)
