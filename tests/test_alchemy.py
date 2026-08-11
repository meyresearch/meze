import pytest
from pathlib import Path
from meze import Meze, Ligand, AlchemicalMezeRecipe
from meze.sofra import AlchemicalSofra

DATA = Path(__file__).parent.parent / "tests" / "data"


def _unbound_meze(ligand_file, name):
    ligand = Ligand(
        file=ligand_file,
        name=name,
        charge=-1,
        parameterised=True,
        residue_name="R9K"
    )
    return Meze.from_files(
        topology=ligand_file,
        coordinates=ligand_file,
        stage="unbound",
        ligand=ligand
    )


@pytest.fixture
def unbound_mezes():
    first = _unbound_meze(str(DATA / "ligands/ligand_11.pdb"), "ligand_11")
    second = _unbound_meze(str(DATA / "ligands/ligand_12.pdb"), "ligand_12")
    return first, second


def test_alchemical_sofra_unbound_happy_path(unbound_mezes, tmp_path):
    first_meze, second_meze = unbound_mezes
    alch = AlchemicalSofra(
        first_meze=first_meze,
        second_meze=second_meze,
        stage="unbound",
        recipe=AlchemicalMezeRecipe(),
        directory=str(tmp_path)
    )
    assert alch.first_molecule.nAtoms() == 31
    assert alch.second_molecule.nAtoms() == 34
    assert alch.bss_base_system.nMolecules() == 1
    assert alch.transformation == "ligand_1~ligand_2"
    assert alch.working_directory == str(
        tmp_path / "ligand_1~ligand_2" / "unbound"
    )


def test_alchemical_sofra_invalid_stage_raises(unbound_mezes, tmp_path):
    first_meze, second_meze = unbound_mezes
    with pytest.raises(ValueError, match="stage must be 'bound' or 'unbound'"):
        AlchemicalSofra(
            first_meze=first_meze,
            second_meze=second_meze,
            stage="bad",
            recipe=AlchemicalMezeRecipe(),
            directory=str(tmp_path)
        )


def test_alchemical_sofra_non_somd_engine_raises(unbound_mezes, tmp_path):
    first_meze, second_meze = unbound_mezes
    with pytest.raises(RuntimeError, match="Currently only supporting SOMD"):
        AlchemicalSofra(
            first_meze=first_meze,
            second_meze=second_meze,
            stage="unbound",
            recipe=AlchemicalMezeRecipe(engine="OpenMM"),
            directory=str(tmp_path)
        )


def test_alchemical_sofra_existing_directory_raises(unbound_mezes, tmp_path):
    first_meze, second_meze = unbound_mezes
    AlchemicalSofra(
        first_meze=first_meze,
        second_meze=second_meze,
        stage="unbound",
        recipe=AlchemicalMezeRecipe(),
        directory=str(tmp_path)
    )
    with pytest.raises(FileExistsError, match="already exists"):
        AlchemicalSofra(
            first_meze=first_meze,
            second_meze=second_meze,
            stage="unbound",
            recipe=AlchemicalMezeRecipe(),
            directory=str(tmp_path),
            overwrite=False
        )
