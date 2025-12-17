from meze import ColdMeze, ColdMezeRecipe
import os
import sys
import json


project_dir = sys.argv[1] 
system_name = "vim2" 
ligand_name = sys.argv[2]

# set ColdMezeRecipe including model (i.e. metal params), ligand(?)
with open(f"{project_dir}/inputs/model_ezaff_recipe.json", "r") as file:
    json_recipe = json.load(file)

json_recipe["path_to_engine"] = os.path.join(
    os.environ["PMEMDHOME"], "bin", "pmemd.cuda"        
)

cold_meze = ColdMeze.from_files(
    recipe=ColdMezeRecipe(**json_recipe),
    pdb_file=f"{project_dir}/inputs/protein/vim2.pdb"
)

cold_meze_with_lig = cold_meze.add_ligand(
    ligand_file=f"{project_dir}/inputs/ligands/{ligand_name}.pdb",
    ligand_charge=-1
)

#TODO make non standard res a union of Ligand and List[Ligand]
cold_system = cold_meze_with_lig.add_non_standard_residue(
    file=f"{project_dir}/inputs/protein/MOH.pdb",
    charge=-1,
    atom_type="amber"
)

cold_complex = cold_system.add_non_standard_residue(
    file=f"{project_dir}/inputs/protein/wat_h.pdb",
    atom_type="amber"
)

