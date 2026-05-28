from meze import ColdMeze
import sys
import json
from pathlib import Path

# REPLACE THESE:
REPO_ROOT = Path().resolve()
EXAMPLES_DIR = REPO_ROOT / "examples"
DATA_DIR = REPO_ROOT / "data"

project_dir = DATA_DIR

# # WITH:
# project_dir = "" # PATH TO FILES
project_dir = "/Users/af25016/Downloads/53/"

system_name = "l1"
ligand_name = "ligand_53"

prepared_meze = ColdMeze.from_files(
    pdb_file=f"{project_dir}/L1_1SML_OH_EZAFF.pdb", # MCPB.py input pdb file
    mcpbpy_input_file=f"{project_dir}/mcpb.in", # MCPB.py input file
    parameterisation_directory=project_dir, # directory to that specific ligand with gaussian outputs
    disulfide_bridges=[{"resid1": 217, "resid2": 245}],
    non_standard_residues={"MOH": {"charge": -1, "atom_type": "amber"},
                           "DOH": {"charge": -1, "atom_type": "amber"}}, 
)
print(prepared_meze)

prepared_complex = prepared_meze.add_ligand(
    ligand_file=f"{project_dir}/RBX.mol2",
    ligand_charge=-1,
    name="RBX",
    residue_name="RBX"
)


bonded_meze = prepared_complex.build_empirical_bonds()

# run once without chgfix_resids
resp_charged_meze = bonded_meze.build_resp_charges(fix_ligand_charge=False) 
resp_charged_meze.save(
     filename=f"{resp_charged_meze.parameterisation_directory}/{ligand_name}_meze_resp_charges"
)

# then fix ligand charge, making a new directory
fixed_ligand_charges_meze = bonded_meze.build_resp_charges(fix_ligand_charge=True) 
fixed_ligand_charges_meze.save(
     filename=f"{fixed_ligand_charges_meze.parameterisation_directory}/{ligand_name}_meze_fixed_ligand_charges"
)

solvated_fixed_charges = fixed_ligand_charges_meze.add_water(
     directory=fixed_ligand_charges_meze.parameterisation_directory,
     mcpbpy_tleap_file=fixed_ligand_charges_meze.tleap_input_file
)
pickled_file = solvated_fixed_charges.save(
     filename=f"{solvated_fixed_charges.parameterisation_directory}/{ligand_name}_fixed_charges_solvated"
)

solvated_fixed_charges.add_to_sofra(
    key=ligand_name,
    filename=f"{project_dir}/inputs/hybrid_model/protein/{system_name}/model_ezaff_sofra.json",
    pickle_file=pickled_file
)




