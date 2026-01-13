from meze import ColdMeze, ColdMezeRecipe
import os
import sys
import json


project_dir = sys.argv[1] 
system_name = "vim2" 
ligand_name = sys.argv[2]

# set ColdMezeRecipe including model (i.e. metal params), ligand(?)
with open(f"{project_dir}/inputs/hybrid_model/model_ezaff_recipe.json", "r") as file:
    json_recipe = json.load(file)

# json_recipe["path_to_engine"] = os.path.join(
#     os.environ["PMEMDHOME"], "bin", "pmemd.cuda"        
# )

cold_meze = ColdMeze.from_files(
    recipe=ColdMezeRecipe(**json_recipe),
    pdb_file=f"{project_dir}/inputs/hybrid_model/protein/vim2.pdb"
)

cold_meze_with_lig = cold_meze.add_ligand(
    ligand_file=f"{project_dir}/inputs/hybrid_model/ligands/{ligand_name}.pdb",
    ligand_charge=-1,
    name="MOL"
)

#TODO make non standard res a union of Ligand and List[Ligand]
cold_system = cold_meze_with_lig.add_non_standard_residue(
    file=f"{project_dir}/inputs/hybrid_model/protein/MOH.pdb",
    charge=-1,
    atom_type="amber"
)

cold_complex = cold_system.add_xtal_water( 
    file=f"{project_dir}/inputs/hybrid_model/protein/wat_h.pdb",
)

parameterised_ligand = cold_complex.ligand.parameterise(
    path=f"{project_dir}/inputs/hybrid_model/ligands/{ligand_name}/",
)

parameterised_hydroxide = cold_complex.non_standard_residue.parameterise(
    path=f"{project_dir}/inputs/hybrid_model/ligands/{ligand_name}",
    atom_type="amber",
    residue_name="MOH"
)

cold_complex.prepare_metals_for_ezaff(
    path=f"{project_dir}/inputs/hybrid_model/ligands/{ligand_name}/"
)

prepared_complex = cold_complex.write_complex(
    path=f"{project_dir}/inputs/hybrid_model/ligands/{ligand_name}/",
    ligand_name=ligand_name
)

# write MCPB.py input file with json input options

# run step 1 of MCPB.py

# write out Gaussian input scripts for RESP calculation (fix scripts)


