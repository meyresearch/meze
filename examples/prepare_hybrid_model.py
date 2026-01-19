from meze import ColdMeze, ColdMezeRecipe
import os
import sys
import json


project_dir = sys.argv[1] 
system_name = "vim2" 
ligand_name = sys.argv[2]

# set ColdMezeRecipe including model (i.e. metal params), ligand(?)
with open(f"{project_dir}/inputs/hybrid_model/protein/{system_name}/model_ezaff_recipe.json", "r") as file:
    json_recipe = json.load(file)

# json_recipe["path_to_engine"] = os.path.join(
#     os.environ["PMEMDHOME"], "bin", "pmemd.cuda"        
# )

cold_meze = ColdMeze.from_files(
    recipe=ColdMezeRecipe(**json_recipe),
    pdb_file=f"{project_dir}/inputs/hybrid_model/protein/{system_name}/vim2.fixed.pdb",
    non_standard_residues={"MOH": {"charge": -1, "atom_type": "amber"}},
)


ligand_file = f"{project_dir}/inputs/hybrid_model/ligands/{system_name}/{ligand_name}/{ligand_name}.pdb"
cold_complex = cold_meze.add_ligand(
    ligand_file=ligand_file,
    ligand_charge=-1,
    name="MOL"
)

input_directory = f"{project_dir}/inputs/hybrid_model/protein/{system_name}/{ligand_name}/"

prepared_complex = cold_complex.prepare_mcpb_system(directory=input_directory,
                                                    ligand_name=ligand_name)

# run step 1 of MCPB.py
prepared_complex.run_mcpb_step_1(ligand_name=ligand_name)

# write out Gaussian input scripts for RESP calculation (fix scripts)


