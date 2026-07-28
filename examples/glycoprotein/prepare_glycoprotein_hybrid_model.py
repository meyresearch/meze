from meze import ColdMeze, ColdMezeRecipe
import json
import MDAnalysis as mda
import glob
from pathlib import Path

system_name = "mII"
ligand_name = "m5g0"

project_dir = Path(__file__).parent / "data" 

with open(f"{project_dir}/inputs/hybrid_model/protein/{system_name}/model_ezaff_recipe.json", "r") as file:
    json_recipe = json.load(file)


ligand_file = f"{project_dir}/inputs/hybrid_model/ligands/{system_name}/{ligand_name}/M5G0_conformer1_only_substrate.pdb"

# split the ligand into fragments:
u = mda.Universe(ligand_file)
for residue in u.residues:
    name = residue.resname
    resid = residue.resid
    ag = u.select_atoms(f"resid {resid}")
    ag.write(f"{project_dir}/inputs/hybrid_model/ligands/{system_name}/{ligand_name}/{name}_{resid}.pdb")

all_ligand_files = sorted(glob.glob(
    f"{project_dir}/inputs/hybrid_model/ligands/{system_name}/{ligand_name}/*.pdb"
))

non_standard_residue_files = [file for file in all_ligand_files if "0MA_5" not in file and "M5G0" not in file]

# all others are loaded as non-standard residues
cold_meze = ColdMeze.from_files(
    recipe=ColdMezeRecipe(**json_recipe),
    pdb_file=f"{project_dir}/inputs/hybrid_model/protein/{system_name}/M5G0_conformer1_glycoprotein_ZN_only.pdb"
)
# treat 0MA_5 as the ligand
ligand_0MA_5 = f"{project_dir}/inputs/hybrid_model/ligands/{system_name}/{ligand_name}/0MA_5.pdb"

cold_complex = cold_meze.add_ligand(
    ligand_file=ligand_0MA_5,
    ligand_charge=0,
    name="MA5"
)

