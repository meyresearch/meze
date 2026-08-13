import os
import pytest
import json
import BioSimSpace as bss
from unittest.mock import MagicMock
from meze import (
    ColdMeze,
    ColdMezeRecipe,
    HotMeze,
    HotMezeRecipe,
    QuantumMeze,
    ColdQuantumMeze,
    HotQuantumMeze,
    MezeRecipe
)
from pathlib import Path

DATA = Path(__file__).parent.parent / "tests" / "data"


@pytest.fixture(scope="session")
def vim2_recipe_json():
    with open(DATA / "vim2/model_0_recipe.json") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def l1_recipe_json():
    with open(DATA / "l1/model_0_recipe.json") as f:
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
        recipe=ColdMezeRecipe(**vim2_recipe_json),
        ligand_resname="MOL"
    )


@pytest.fixture(scope="session")
def vim2_hot_meze(vim2_recipe_json):
    return HotMeze.from_files(
        topology=str(DATA / "vim2/vim2_complex.prmtop"),
        coordinates=str(DATA / "vim2/vim2_complex.inpcrd"),
        recipe=HotMezeRecipe(**vim2_recipe_json),
        ligand_resname="MOL"
    )


@pytest.fixture(scope="session")
def vim2_quantum_meze(vim2_recipe_json):
    return QuantumMeze.from_files(
        topology=str(DATA / "vim2/vim2_complex.prmtop"),
        coordinates=str(DATA / "vim2/vim2_complex.inpcrd"),
        recipe=MezeRecipe(**vim2_recipe_json),
    )


@pytest.fixture(scope="session")
def vim2_cold_quantum_meze(vim2_recipe_json):
    return QuantumMeze.from_files(
        topology=str(DATA / "vim2/vim2_complex.prmtop"),
        coordinates=str(DATA / "vim2/vim2_complex.inpcrd"),
        recipe=ColdQuantumMeze(vim2_recipe_json),
    )


@pytest.fixture(scope="session")
def vim2_hot_quantum_meze(vim2_recipe_json):
    return QuantumMeze.from_files(
        topology=str(DATA / "vim2/vim2_complex.prmtop"),
        coordinates=str(DATA / "vim2/vim2_complex.inpcrd"),
        recipe=HotQuantumMeze(**vim2_recipe_json),
    )


@pytest.fixture
def mock_amber_process():
    processes = []

    def _side_effect(*args, **kwargs):
        system = kwargs.get("system")
        work_dir = kwargs.get("work_dir")
        name = kwargs.get("name")
        bss.IO.saveMolecules(
            os.path.join(work_dir, name),
            system=system,
            fileformat=["prm7", "rst7"]
        )
        process = MagicMock()
        process._config_file = os.path.join(work_dir, f"{name}.cfg")
        process.isError.return_value = False
        process.getSystem.return_value = system
        processes.append(process)
        return process

    _side_effect.processes = processes
    return _side_effect


@pytest.fixture
def mock_amber_process_error():
    def _side_effect(*args, **kwargs):
        system = kwargs.get("system")
        work_dir = kwargs.get("work_dir")
        name = kwargs.get("name")
        bss.IO.saveMolecules(
            os.path.join(work_dir, name),
            system=system,
            fileformat=["prm7", "rst7"]
        )
        process = MagicMock()
        process.isError.return_value = True
        process.getSystem.return_value = system
        process.getStderr.return_value = ["some amber error line"]
        return process

    return _side_effect
