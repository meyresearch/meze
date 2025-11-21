from meze import ColdMeze, ColdMezeRecipe
import os
import json

system_name = "vim2"
ligand_name = "ligand_11"

project_dir = f"/Users/af25016/projects/meze/data/"

# set ColdMezeRecipe including model (i.e. metal params), ligand(?)
with open(f"{project_dir}/model_0_recipe.json", "r") as file:
    json_recipe = json.load(file)

json_recipe["path_to_engine"] = os.path.join(
        os.environ["AMBERHOME"], "bin", "sander"
    )

input_dir = f"{project_dir}/protein/solvate_{ligand_name}_bound/"

cold_meze = ColdMeze.from_files(
    topology=f"{input_dir}/{ligand_name}_complex_solv.prmtop",
    coordinates=f"{input_dir}/{ligand_name}_complex_solv.inpcrd",
    recipe=ColdMezeRecipe(**json_recipe)
)

print(cold_meze.recipe)
                    
equil_dir = os.path.join(project_dir, "equilibration", f"{ligand_name}")
os.makedirs(equil_dir, exist_ok=True)

print("Minimising")

minimised_meze = cold_meze.minimise(
    process_name="01_min",
    workdir=equil_dir,
    position_restraints="solute",
    max_cycles=10,
    is_gpu=True
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

print("07 - Continue restraint weight on backbone atoms (and metal coordination)")

continue_lowering = relax_backbone.pressurise(
     restart=True,
     process_name="07_continue",
     workdir=equil_dir,
     position_restraints="backbone",
     timestep=0.001,
     restraint_weight=0.1
)
