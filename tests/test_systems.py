import os
import tempfile
import pytest
from meze import ColdMeze, ColdMezeRecipe, Ligand
from meze.sofra import Meze
from pathlib import Path


DATA = Path(__file__).parent.parent / "tests" / "data"


def test_missing_topology_file_raises(vim2_recipe_json):
    with pytest.raises(FileNotFoundError):
        ColdMeze.from_files(
            topology="nonexistent/topology.prm7",
            coordinates=str(DATA / "vim2/vim2.fixed.pdb"),
            recipe=ColdMezeRecipe(**vim2_recipe_json)
        )


def test_missing_coordinates_file_raises(vim2_recipe_json):
    with pytest.raises(FileNotFoundError):
        ColdMeze.from_files(
            coordinates="nonexistent/coordinates.rst7",
            topology=str(DATA / "vim2/vim2.fixed.pdb"),
            recipe=ColdMezeRecipe(**vim2_recipe_json)
        )


def test_check_file_exists_real_file_ok(tmp_path):
    real_file = tmp_path / "topology.prm7"
    real_file.write_text("dummy")
    assert Meze._check_file_exists(str(real_file)) is None


def test_get_coordinate_fileformat_rst7():
    assert Meze._get_coordinate_fileformat(".rst7") == "RESTRT"


@pytest.mark.parametrize("extension", [".pdb", ".inpcrd", ""])
def test_get_coordinate_fileformat_other_extensions(extension):
    assert Meze._get_coordinate_fileformat(extension) is None


def test_coordinating_residues_identifies_zn_ligands(vim2_cold_meze):
    m = vim2_cold_meze
    assert len(m.metals) == 2
    assert list(m.metal_resids) == [233, 234]
    assert list(m.metal_atomids) == [3450, 3451]
    assert set(m.coordinating_residues.keys()) == {3450, 3451}
    for ligands in m.coordinating_residues.values():
        assert len(ligands) == 4
    first_resids = m.coordinating_residues[3450].resids
    assert set(first_resids) == {83, 85, 148, 235}
    second_resids = m.coordinating_residues[3451].resids
    assert set(second_resids) == {87, 167, 209, 235}


def test_non_standard_residues(vim2_cold_meze):
    assert vim2_cold_meze.non_standard_residues == {
        "MOH": {"charge": -1, "atom_type": "amber"}
    }


def test_non_standard_residues_missing_keys_raises(vim2_recipe_json):
    with pytest.raises(ValueError, match="must have 'charge' and 'atom_type'"):
        ColdMeze.from_files(
            recipe=ColdMezeRecipe(**vim2_recipe_json),
            pdb_file=str(DATA / "vim2/vim2.fixed.pdb"),
            non_standard_residues={"MOH": {"charge": -1}},
        )


def test_non_standard_residues_non_int_charge_raises(vim2_recipe_json):
    with pytest.raises(ValueError, match="invalid 'charge'"):
        ColdMeze.from_files(
            recipe=ColdMezeRecipe(**vim2_recipe_json),
            pdb_file=str(DATA / "vim2/vim2.fixed.pdb"),
            non_standard_residues={
                "MOH": {"charge": "bad", "atom_type": "amber"}
            },
        )


def test_add_ligand(vim2_cold_meze):
    ligand = vim2_cold_meze.ligand
    assert ligand.file == [str(DATA / "ligands/ligand_11.pdb")]
    assert ligand.name == "ligand_11"
    assert ligand.charge == -1
    assert ligand.atom_type == "gaff2"
    assert ligand.parameterised is False
    assert ligand.frcmod_file is None
    assert ligand.residue_name == "R9K"
    assert ligand.topology is None
    assert ligand.coordinates is None
    assert ligand.system.nMolecules() == 1


def test_set_metal_raises_for_absent_metal(vim2_recipe_json):
    bad_recipe = ColdMezeRecipe(**{**vim2_recipe_json, "metal": "FE"})
    with pytest.raises(ValueError, match="No atoms found for metal"):
        ColdMeze.from_files(
            recipe=bad_recipe,
            pdb_file=str(DATA / "vim2/vim2.fixed.pdb"),
        )


def test_select_crystal_waters(vim2_cold_meze):
    assert set(vim2_cold_meze.crystal_waters.resnames) == {"WAT"}
    assert vim2_cold_meze.crystal_waters.n_residues == 182
