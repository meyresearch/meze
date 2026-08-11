import pytest
from meze import Ligand
from pathlib import Path

DATA = Path(__file__).parent.parent / "tests" / "data"


def test_ligand_happy_path_construction():
    ligand = Ligand(
        file=str(DATA / "ligands/ligand_11.pdb"),
        charge=-1,
        name="ligand_11"
    )
    assert ligand.file == [str(DATA / "ligands/ligand_11.pdb")]
    assert ligand.charge == -1.0
    assert isinstance(ligand.charge, float)
    assert ligand.residue_name == "R9K"
    assert ligand.atom_type == "gaff2"
    assert ligand.parameterised is False
    assert ligand.system.nMolecules() == 1
    assert ligand.system.nAtoms() == 31


def test_validate_file_str_wrapped_in_list():
    assert Ligand._validate_file("ligand_11.pdb") == ["ligand_11.pdb"]


def test_validate_file_list():
    files = ["ligand_11.pdb", "ligand_11.mol2"]
    assert Ligand._validate_file(files) == files


def test_validate_file_too_many_raises():
    with pytest.raises(ValueError, match="Too many values"):
        Ligand._validate_file(["a.pdb", "b.pdb", "c.pdb"])


def test_validate_file_wrong_type_raises():
    with pytest.raises(TypeError, match="Expected str or list"):
        Ligand._validate_file(123)


def test_check_files_exist_passes_for_real_files(tmp_path):
    file_1 = tmp_path / "a.pdb"
    file_2 = tmp_path / "b.pdb"
    file_1.write_text("dummy")
    file_2.write_text("dummy")

    Ligand._check_files_exist([str(file_1), str(file_2)])


def test_check_files_exist_missing_file_raises():
    with pytest.raises(FileNotFoundError, match="Ligand file not found"):
        Ligand._check_files_exist(["/nonexistent/ligand.pdb"])


def test_validate_charge_float():
    assert Ligand._validate_charge(1.0) == 1.0


def test_validate_charge_int_coerced_to_float():
    charge = Ligand._validate_charge(1)
    assert charge == 1.0
    assert isinstance(charge, float)


def test_validate_charge_string_coerced_to_float():
    charge = Ligand._validate_charge("-1")
    assert charge == -1.0
    assert isinstance(charge, float)


def test_validate_charge_invalid_raises():
    with pytest.raises(TypeError, match="must be an integer or float"):
        Ligand._validate_charge("abc")


def test_infer_ligand_name_uses_file_stem():
    with pytest.warns(UserWarning, match="inferring from file name"):
        name = Ligand._infer_ligand_name(["some/dir/ligand_11.pdb"])
    assert name == "ligand_11"


def test_more_than_one_residue_warning(caplog):
    Ligand(
        file=str(DATA / "vim2/water.pdb"),
        charge=-1,
        name="ligand_11"
    )
    assert "Found multiple residues in ligand file" in caplog.text
