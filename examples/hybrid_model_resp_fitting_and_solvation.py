from meze import ColdMeze
import sys
import json

project_dir = sys.argv[1] 
system_name = sys.argv[2]
ligand_name = sys.argv[3]


with open(f"{project_dir}/inputs/hybrid_model/protein/{system_name}/model_ezaff_sofra.json", "r") as file:
     json_recipe = json.load(file)

parameterisation_directory = json_recipe[ligand_name]["parameterisation_directory"]

prepared_meze = ColdMeze.load(
    filename=f"{parameterisation_directory}/{ligand_name}_meze.pkl"
)
print(prepared_meze)

prepared_meze.build_empirical_bonds()

# run once without chgfix_resids
resp_charged_meze = prepared_meze.build_resp_charges(fix_ligand_charge=False) 
resp_charged_meze.save(
     filename=f"{resp_charged_meze.parameterisation_directory}/{ligand_name}_meze_resp_charges"
)

# then fix ligand charge, making a new directory
fixed_ligand_charges_meze = prepared_meze.build_resp_charges(fix_ligand_charge=True) 
fixed_ligand_charges_meze.save(
     filename=f"{fixed_ligand_charges_meze.parameterisation_directory}/{ligand_name}_meze_fixed_ligand_charges"
)

solvated_fixed_charges = fixed_ligand_charges_meze.add_water(
     directory=fixed_ligand_charges_meze.parameterisation_directory,
     mcpbpy_tleap_file=fixed_ligand_charges_meze.tleap_input_file
)
solvated_fixed_charges.save(
     filename=f"{solvated_fixed_charges.parameterisation_directory}/{ligand_name}_fixed_charges_solvated"
)

solvated_fixed_charges.add_to_sofra(
    key=ligand_name,
    filename=f"{project_dir}/inputs/hybrid_model/protein/{system_name}/model_ezaff_sofra.json"
)




