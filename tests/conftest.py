import pytest
import json
from meze import ColdMeze, ColdMezeRecipe
from pathlib import Path

DATA = Path(__file__).parent.parent / "tests" / "data"


@pytest.fixture(scope="session")
def vim2_recipe_json():
    with open(DATA / "vim2/model_0_recipe.json") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def vim2_cold_meze(vim2_recipe_json):
    meze = ColdMeze.from_files(
        recipe=ColdMezeRecipe(**vim2_recipe_json),
        pdb_file=str(DATA / "vim2/vim2.fixed.pdb"),
        non_standard_residues={"MOH": {"charge": -1, "atom_type": "amber"}}
    )
    return meze.add_ligand(
        ligand_file=str(DATA / "ligands/ligand_11.pdb"),
        name="ligand_11",
        ligand_charge=-1
    )


@pytest.fixture(scope="session")
def vim2_top_and_coord(vim2_recipe_json):
    return ColdMeze.from_files(
        topology=str(DATA / "vim2/vim2_complex.prmtop"),
        coordinates=str(DATA / "vim2/vim2_complex.inpcrd"),
        recipe=ColdMezeRecipe(**vim2_recipe_json)
    )
