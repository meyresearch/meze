from meze import ColdMeze, ColdMezeRecipe
import os
import sys
import json

project_dir = sys.argv[1] 
system_name = sys.argv[2]
ligand_name = sys.argv[3]

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

output = f"{project_dir}/outputs/hybrid_model/{system_name}/{ligand_name}/"

mcpb_system = cold_complex.prepare_mcpb_system(directory=output,
                                               ligand_name=ligand_name)

scratch_dir = os.path.join(mcpb_system.recipe.parameterisation_directory, "scratch")
os.makedirs(scratch_dir, exist_ok=True)

prepared_complex = mcpb_system.prepare_resp_calculation(
    ligand_name=ligand_name,
    sbatch_options={
        "nodes": 1,
        "partition": "short",
        "ntasks-per-node": 8,
        "mem": "24GB",
        "account": "",
    },
    additional_lines=[
        "module load apps/gaussian\n",
        "LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH\n",
        f"export GAUSS_SCRDIR={scratch_dir}\n",
    ]
)

prepared_complex.save(
    filename=f"{prepared_complex.recipe.parameterisation_directory}/{ligand_name}_meze"
)

# Save updated recipe
prepared_complex.recipe.to_json(
    f"{project_dir}/inputs/hybrid_model/protein/{system_name}/model_ezaff_recipe.json"
)

# Show pretty print
print(prepared_complex)

