from meze import ColdMeze, ColdMezeRecipe
import os
import sys
import json

project_dir = "/Users/af25016/Downloads/53/"
system_name = "l1"
ligand_name = "ligand_53"

# set ColdMezeRecipe including model (i.e. metal params), ligand(?)
with open("/Users/af25016/projects/meze/data/inputs/hybrid_model/protein/l1/model_ezaff_recipe.json", "r") as file:
    json_recipe = json.load(file)

# json_recipe["path_to_engine"] = os.path.join(
#     os.environ["PMEMDHOME"], "bin", "pmemd.cuda"        
# )

cold_meze = ColdMeze.from_files(
    recipe=ColdMezeRecipe(**json_recipe),
    pdb_file=f"{project_dir}/L1_1SML_OH_EZAFF.pdb",
    disulfide_bridges=[{"resid1": 217, "resid2": 245}, 
                       {"resid1": 487, "resid2": 515}],
    non_standard_residues={"MOH": {"charge": -1, "atom_type": "amber"},
                           "DOH": {"charge": -1, "atom_type": "amber"}}, 
)

cold_complex = cold_meze.add_ligand(
    ligand_file=f"{project_dir}/RBX.mol2",
    ligand_charge=-1,
    name="RBX",
    residue_name="RBX",
    parameterised=True,
    frcmod_file=f"{project_dir}/RBX.frcmod"
)

output = f"{project_dir}/outputs/hybrid_model/{system_name}/{ligand_name}/"

mcpb_system = cold_complex.prepare_mcpb_system(directory=output,
                                               ligand_name="RBX",
                                               ligand_file_name="RBX")

scratch_dir = os.path.join(mcpb_system.parameterisation_directory, "scratch")
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

pickled_file = prepared_complex.save(
    filename=f"{prepared_complex.parameterisation_directory}/{ligand_name}_meze"
)

# Save updated recipe -> separate into ligand recipe? 
prepared_complex.recipe.to_json(
    f"{project_dir}/inputs/hybrid_model/protein/{system_name}/model_ezaff_recipe.json"
)

prepared_complex.add_to_sofra(
    key=ligand_name,
    filename=f"{project_dir}/inputs/hybrid_model/protein/{system_name}/model_ezaff_sofra.json",
    pickle_file=pickled_file
)


# Show pretty print
print(prepared_complex)

