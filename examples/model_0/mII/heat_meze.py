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

equil_dir = os.path.join(project_dir, "equilibration", "model_0", system_name, f"{ligand_name}")

os.makedirs(equil_dir, exist_ok=True)

print("Minimising")

minimised_meze = solvated_meze.minimise(
    process_name="01_min",
    workdir=equil_dir,
    position_restraints="solute",
    max_cycles=5000,
    is_gpu=False,
    additional_positional_restraints={"resnames": ["0YB", "ROH", "4YA", "0MA", "2MA", "VMA", "VMB", "NLN"]}
)

print("02 - Heating with restrained solute")

hot_meze = minimised_meze.heat(
    process_name="02_heat",
    workdir=equil_dir,
    position_restraints="solute",
    timestep=0.001,
    start_temperature=100,
    end_temperature=300,
        additional_positional_restraints={"resnames": ["0YB", "ROH", "4YA", "0MA", "2MA", "VMA", "VMB", "NLN"]}   
) 

print("03 - Constant temperature with restrained solute")

relax_meze = hot_meze.heat(
    restart=True,
    process_name="03_relax",
    workdir=equil_dir,
    position_restraints="solute",
    timestep=0.001,
    additional_positional_restraints={"resnames": ["0YB", "ROH", "4YA", "0MA", "2MA", "VMA", "VMB", "NLN"]}
)

print("04 - Add pressure with restrained solute")

pressure_meze = relax_meze.pressurise(
     restart=True,
     process_name="04_pressure",
     workdir=equil_dir, 
     position_restraints="solute",
     timestep=0.001,
     additional_restraints={"resnames": ["0YB", "ROH", "4YA", "0MA", "2MA", "VMA", "VMB", "NLN"]}
)

print("05 - Start lowering restraint weight on  solute")

lower_restraint = pressure_meze.pressurise(
     restart=True,
     process_name="05_lower",
     workdir=equil_dir,
     position_restraints="solute",
     timestep=0.001,
     restraint_weight=10.0,
     additional_restraints={"resnames": ["0YB", "ROH", "4YA", "0MA", "2MA", "VMA", "VMB", "NLN"]}
)

print("06 - Only restrain backbone atoms (and metal coordination)")

relax_backbone = lower_restraint.pressurise(
     restart=True,
     process_name="06_relax",
     workdir=equil_dir,
     position_restraints="backbone",
     timestep=0.001,
     restraint_weight=10.0,
     additional_restraints={"resnames": ["0YB", "ROH", "4YA", "0MA", "2MA", "VMA", "VMB", "NLN"]}

)

print("07 - Reduce restraint")

reduce_restraint = relax_backbone.pressurise(
     restart=True,
     process_name="07_reduce",
     workdir=equil_dir,
     timestep=0.001,
     position_restraints="metal-coordination",
     restraint_weight=1.0,
     additional_restraints={"resnames": ["0YB", "ROH", "4YA", "0MA", "2MA", "VMA", "VMB", "NLN"]}
)


