from meze import ColdMeze, ColdMezeRecipe
import json
import os

system_name = "vim2"
bridging_water = "hydroxide"
ligand_name = "ligand_11"

project_dir = f"/Users/af25016/projects/meze/data/"

# set ColdMezeRecipe including model (i.e. metal params), ligand(?)
with open(f"{project_dir}/model_0_recipe.json", "r") as file:
    json_recipe = json.load(file)

json_recipe["path_to_engine"] = os.path.join(
        os.environ["AMBERHOME"], "bin", "sander"
    )
# load in protein files into ColdMeze
cold_meze = ColdMeze.from_files(
    recipe=ColdMezeRecipe(**json_recipe),
    pdb_file=f"{project_dir}/protein/vim2.fixed.pdb",
    
)

cold_meze_with_lig = cold_meze.add_ligand(
    ligand_file=f"{project_dir}/ligands/{ligand_name}.pdb",
    ligand_charge=-1
)

#TODO make non standard res a union of Ligand and List[Ligand]
cold_system = cold_meze_with_lig.add_non_standard_residue(
    file=f"{project_dir}/protein/MOH.pdb",
    charge=-1,
    atom_type="amber"
)

print(cold_system)

# solvate <-- write solvate.py script
#TODO: put solvation options into MezeRecipe
solvate_dir = f"{project_dir}/protein/solvate_{ligand_name}_bound/"


solvated_meze = cold_system.add_water(directory=solvate_dir)

print(solvated_meze)

# heat meze
equil_dir = os.path.join(project_dir, "equilibration", f"{ligand_name}")
os.makedirs(equil_dir, exist_ok=True)

print("Minimising")

minimised_meze = solvated_meze.minimise(
    process_name="01_min",
    workdir=equil_dir,
    position_restraints="solute",
    max_cycles=50,
    is_gpu=False
)

print("02 - Heating with restrained solute")

hot_meze = minimised_meze.heat(
    process_name="02_heat",
    workdir=equil_dir,
    position_restraints="solute",
    timestep=0.001,
    runtime=20,
    start_temperature=100,
    end_temperature=300,
    is_gpu=False            
) 

# run production