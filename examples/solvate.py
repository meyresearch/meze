from meze import ColdMeze, ColdMezeRecipe
import os
import sys
import json

project_dir = sys.argv[1] 
system_name = sys.argv[2] 
ligand_name = sys.argv[3]

with open(f"{project_dir}/inputs/model_0/protein/{system_name}/model_0_recipe.json", "r") as file:
    json_recipe = json.load(file)

# json_recipe["path_to_engine"] = os.path.join(
#     os.environ["PMEMDHOME"], "bin", "pmemd.cuda"        
# )

cold_meze = ColdMeze.from_files(
    recipe=ColdMezeRecipe(**json_recipe),
    pdb_file=f"{project_dir}/inputs/model_0/protein/{system_name}/{system_name}.fixed.pdb",
    disulfide_bridges=[{
        "resid1": 217,
        "resid2": 245
    }, {
        "resid1": 486,
        "resid2": 514
    }]
)

cold_meze_with_lig = cold_meze.add_ligand(
    ligand_file=f"{project_dir}/inputs/model_0/ligands/{system_name}/{ligand_name}.pdb",
    ligand_charge=-1
)

cold_system = cold_meze_with_lig.add_non_standard_residue(
    files=[f"{project_dir}/inputs/model_0/protein/{system_name}/MOH.pdb",
           f"{project_dir}/inputs/model_0/protein/{system_name}/DOH.pdb"],
    charges=[-1, -1], 
    atom_types=["amber", "amber"],
    names=["MOH", "DOH"] 
)

print(cold_system)

#TODO: put solvation options into MezeRecipe
solvate_dir = f"{project_dir}/inputs/model_0/protein/{system_name}/solvate_{ligand_name}_bound/"
solvated_meze = cold_system.add_water(directory=solvate_dir)

