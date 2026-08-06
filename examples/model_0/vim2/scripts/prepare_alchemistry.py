from meze import (
    Meze, 
    HotMeze, 
    HotMezeRecipe, 
    Sofra, 
    AlchemicalSofra, 
    AlchemicalMezeRecipe
)
import os
import json
import sys
from pathlib import Path
import glob


REPO_ROOT = Path().resolve()
EXAMPLES_DIR = REPO_ROOT / "examples"
DATA_DIR = REPO_ROOT / "data"
project_dir = DATA_DIR

system_name = "vim2"

ligand_1 = "ligand_11"
ligand_2 = "ligand_12"
repeat = 1

with open(f"{project_dir}/inputs/model_0/protein/{system_name}/model_0_recipe.json", "r") as file:
    json_recipe = json.load(file)

json_recipe["path_to_engine"] = os.path.join(
        os.environ["PMEMDHOME"], "bin", "pmemd.cuda"
)

bound_lig_1_equil_dir = f"{project_dir}/equilibration/model_0/{system_name}/bound/{ligand_1}/repeat_{repeat}/"

bound_ligand_1_meze = HotMeze.from_files(
    recipe=HotMezeRecipe(**json_recipe),
    topology=f"{bound_lig_1_equil_dir}/08_free/next.prm7",
    coordinates=f"{bound_lig_1_equil_dir}/08_free/next.rst7",
    restraint_file=f"{bound_lig_1_equil_dir}/08_free/restraints.RST",
    ligand_resname="MOL"
)

bound_lig_2_equil_dir = f"{project_dir}/equilibration/model_0/{system_name}/bound/{ligand_2}/repeat_{repeat}/"

bound_ligand_2_meze = HotMeze.from_files(
    recipe=HotMezeRecipe(**json_recipe),
    topology=f"{bound_lig_2_equil_dir}/08_free/next.prm7",
    coordinates=f"{bound_lig_2_equil_dir}/08_free/next.rst7",
    restraint_file=f"{bound_lig_2_equil_dir}/08_free/restraints.RST",
    ligand_resname="MOL"
)

unbound_lig_1_equil_dir = f"{project_dir}/equilibration/model_0/{system_name}/unbound/{ligand_1}/repeat_{repeat}/"

unbound_ligand_1_meze = HotMeze.from_files(
    recipe=HotMezeRecipe(**json_recipe),
    topology=f"{unbound_lig_1_equil_dir}/06_free/next.prm7",
    coordinates=f"{unbound_lig_1_equil_dir}/06_free/next.rst7",
    stage="unbound",
    ligand_resname="MOL"
)

unbound_lig_2_equil_dir = f"{project_dir}/equilibration/model_0/{system_name}/unbound/{ligand_2}/repeat_{repeat}/"

unbound_ligand_2_meze = HotMeze.from_files(
    recipe=HotMezeRecipe(**json_recipe),
    topology=f"{unbound_lig_2_equil_dir}/06_free/next.prm7",
    coordinates=f"{unbound_lig_2_equil_dir}/06_free/next.rst7",
    stage="unbound",
    ligand_resname="MOL"
)

alchemical_recipe = AlchemicalMezeRecipe(
    n_lambdas=3,
    sampling_time=1,
    engine="SOMD"
)

unbound_sofra = AlchemicalSofra(
    first_meze=unbound_ligand_1_meze,
    second_meze=unbound_ligand_2_meze,
    first_name=ligand_1,
    second_name=ligand_2,
    stage="unbound",
    directory=f"{project_dir}/outputs/model_0/{system_name}",
    overwrite=True,
    recipe=alchemical_recipe
)

bound_sofra = AlchemicalSofra(
    first_meze=bound_ligand_1_meze,
    second_meze=bound_ligand_2_meze,
    first_name=ligand_1,
    second_name=ligand_2,
    stage="bound",
    directory=f"{project_dir}/outputs/model_0/{system_name}",
    overwrite=True,
    recipe=alchemical_recipe
)

merged_unbound_system = unbound_sofra.setup_alchemistry(
    compute_platform="gpu", 
    debug=True
)

merged_bound_system = bound_sofra.setup_alchemistry(
    compute_platform="gpu", 
    debug=True
)
