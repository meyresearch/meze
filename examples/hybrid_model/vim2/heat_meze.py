from meze import ColdMeze, ColdMezeRecipe
import os
import json
import sys
from pathlib import Path

REPO_ROOT = Path().resolve()
EXAMPLES_DIR = REPO_ROOT / "examples"
DATA_DIR = REPO_ROOT / "data"

system_name = "vim2"

project_dir = DATA_DIR

ligand_name = sys.argv[1]
repeat = sys.argv[2]

# get Sofra which has the information about prepared mezes 
with open(f"{project_dir}/inputs/hybrid_model/protein/{system_name}/model_ezaff_sofra.json", "r") as file:
    json_sofra = json.load(file)

input_dir = json_sofra[ligand_name]["parameterisation_directory"]

solvated_meze = ColdMeze.load(
    filename=f"{input_dir}/{ligand_name}_avg_charges_solv.pkl"
)

solvated_meze.recipe.path_to_engine = os.path.join("AMBERHOME", "bin", "sander")

print(solvated_meze)

equil_dir = os.path.join(project_dir, "equilibration", "hybrid_model", system_name, f"{ligand_name}", f"repeat_{repeat}")

os.makedirs(equil_dir, exist_ok=True)

print("Minimising")

minimised_meze = solvated_meze.minimise(
    process_name="01_min",
    workdir=equil_dir,
    position_restraints="solute",
    max_cycles=5000,
    is_gpu=False
)

print("02 - Heating with restrained solute")

hot_meze = minimised_meze.heat(
    process_name="02_heat",
    workdir=equil_dir,
    position_restraints="solute",
    timestep=0.001,
    start_temperature=100,
    end_temperature=300            
) 

print("03 - Constant temperature with restrained solute")

relax_meze = hot_meze.heat(
    restart=True,
    process_name="03_relax",
    workdir=equil_dir,
    position_restraints="solute",
    timestep=0.001
)

print("04 - Add pressure with restrained solute")

pressure_meze = relax_meze.pressurise(
     restart=True,
     process_name="04_pressure",
     workdir=equil_dir, 
     position_restraints="solute",
     timestep=0.001
)

print("05 - Start lowering restraint weight on  solute")

lower_restraint = pressure_meze.pressurise(
     restart=True,
     process_name="05_lower",
     workdir=equil_dir,
     position_restraints="solute",
     timestep=0.001,
     restraint_weight=10.0
)

print("06 - Only restrain backbone atoms (and metal coordination)")

relax_backbone = lower_restraint.pressurise(
     restart=True,
     process_name="06_relax",
     workdir=equil_dir,
     position_restraints="backbone",
     timestep=0.001,
     restraint_weight=10.0
)

print("07 - Reduce restraint")

reduce_restraint = relax_backbone.pressurise(
     restart=True,
     process_name="07_reduce",
     workdir=equil_dir,
     timestep=0.001,
     position_restraints="metal-coordination",
     restraint_weight=1.0,
)

print("08 - No restraint")

free = reduce_restraint.pressurise(
     restart=True,
     process_name="08_free",
     workdir=equil_dir,
     timestep=0.001,
)

