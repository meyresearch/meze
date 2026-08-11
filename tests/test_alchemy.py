import os
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


def _solvated_unbound_meze(name):
    return Meze.from_files(
        topology=str(DATA / f"ligands/{name}_solv.prmtop"),
        coordinates=str(DATA / f"ligands/{name}_solv.inpcrd"),
        stage="unbound",
        ligand_resname="MOL"
    )


@pytest.fixture
def solvated_alchemical_sofra(tmp_path):
    first_meze = _solvated_unbound_meze("ligand_11")
    second_meze = _solvated_unbound_meze("ligand_12")
    return AlchemicalSofra(
        first_meze=first_meze,
        second_meze=second_meze,
        stage="unbound",
        recipe=AlchemicalMezeRecipe(n_lambdas=3),
        directory=str(tmp_path)
    )


def _bound_meze(topology, coordinates):
    return Meze.from_files(
        topology=topology,
        coordinates=coordinates,
        stage="bound",
        ligand_resname="MOL"
    )


@pytest.fixture
def solvated_bound_alchemical_sofra(tmp_path):
    first_meze = _bound_meze(
        str(DATA / "vim2/vim2_complex.prmtop"),
        str(DATA / "vim2/vim2_complex.inpcrd")
    )
    second_meze = _bound_meze(
        str(DATA / "vim2/ligand_12_complex_solv.prmtop"),
        str(DATA / "vim2/ligand_12_complex_solv.inpcrd")
    )
    return AlchemicalSofra(
        first_meze=first_meze,
        second_meze=second_meze,
        stage="bound",
        recipe=AlchemicalMezeRecipe(n_lambdas=3, model=0),
        directory=str(tmp_path)
    )


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


def test_merge(solvated_alchemical_sofra):
    merged = solvated_alchemical_sofra.merge()
    assert merged.nAtoms() == 34


def test_create_hybrid_molecule(solvated_alchemical_sofra):
    hybrid_system = solvated_alchemical_sofra.create_hybrid_molecule()
    assert hybrid_system.nMolecules() == 772
    assert solvated_alchemical_sofra.merged_molecule is not None
    assert solvated_alchemical_sofra.merged_molecule.nAtoms() == 34


def test_setup_alchemistry(solvated_alchemical_sofra):
    result = solvated_alchemical_sofra.setup_alchemistry()
    assert result.nMolecules() == 772

    working_directory = solvated_alchemical_sofra.working_directory
    lambda_dirs = sorted(os.listdir(working_directory))
    assert lambda_dirs == ["lambda_0.0000", "lambda_0.5000", "lambda_1.0000"]
    for lambda_dir in lambda_dirs:
        assert os.path.isfile(
            os.path.join(working_directory, lambda_dir, "somd.cfg")
        )


def test_merge_bound(solvated_bound_alchemical_sofra):
    merged = solvated_bound_alchemical_sofra.merge()
    assert merged.nAtoms() == 34


def test_create_hybrid_molecule_bound(solvated_bound_alchemical_sofra):
    hybrid_system = solvated_bound_alchemical_sofra.create_hybrid_molecule()
    assert hybrid_system.nMolecules() == 7188
    assert solvated_bound_alchemical_sofra.merged_molecule is not None
    assert solvated_bound_alchemical_sofra.merged_molecule.nAtoms() == 34


def test_setup_alchemistry_bound(solvated_bound_alchemical_sofra, caplog):
    result = solvated_bound_alchemical_sofra.setup_alchemistry()
    assert result.nMolecules() == 7188
    assert (
        "Model 0 bound stage requested without a restraint_file"
        in caplog.text
    )

    working_directory = solvated_bound_alchemical_sofra.working_directory
    lambda_dirs = sorted(os.listdir(working_directory))
    assert lambda_dirs == ["lambda_0.0000", "lambda_0.5000", "lambda_1.0000"]
    for lambda_dir in lambda_dirs:
        assert os.path.isfile(
            os.path.join(working_directory, lambda_dir, "somd.cfg")
        )
