from meze import ColdMeze, ColdMezeRecipe
import os
import sys
import json

project_dir = sys.argv[1] 
system_name = sys.argv[2] 
ligand_name = sys.argv[3]


# set ColdMezeRecipe including model (i.e. metal params), ligand(?)
with open(f"{project_dir}/inputs/model_0/protein/{system_name}/model_0_recipe.json", "r") as file:
    json_recipe = json.load(file)

# json_recipe["path_to_engine"] = os.path.join(
#     os.environ["PMEMDHOME"], "bin", "pmemd.cuda"        
# )

# load in protein files into ColdMeze
cold_meze = ColdMeze.from_files(
    recipe=ColdMezeRecipe(**json_recipe),
    pdb_file=f"{project_dir}/inputs/protein/{system_name}/{system_name}.fixed.pdb"
)

cold_meze_with_lig = cold_meze.add_ligand(
    ligand_file=f"{project_dir}/inputs/ligands/{system_name}/{ligand_name}.pdb",
    ligand_charge=-1
)

#TODO make non standard res a union of Ligand and List[Ligand]
cold_system = cold_meze_with_lig.add_non_standard_residue(
    file=[f"{project_dir}/inputs/protein/{system_name}/MOH.pdb",
          f"{project_dir}/inputs/protein/{system_name}/DOH.pdb"],
    charge=-1,
    atom_type="amber"
)

print(cold_system)

# solvate <-- write solvate.py script
#TODO: put solvation options into MezeRecipe
solvate_dir = f"{project_dir}/inputs/protein/{system_name}/solvate_{ligand_name}_bound/"

solvated_meze = cold_system.add_water(directory=solvate_dir)

