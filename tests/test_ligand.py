import pytest
from meze import Ligand
from pathlib import Path
from unittest.mock import patch
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


def test_more_than_one_residue_warning():
    with pytest.warns(
        UserWarning, match="Found multiple residues in ligand file"
    ):
        Ligand(
            file=str(DATA / "vim2/water.pdb"),
            charge=-1,
            name="ligand_11"
        )


def test_run_antechamber_builds_command_and_strips_du(
        tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    ligand = Ligand(
        file=str(DATA / "ligands/ligand_11.pdb"), charge=-1, name="ligand_11"
    )
    mol2_path = str(tmp_path / "ligand_11.mol2")

    def fake_system(cmd):
        Path(mol2_path).write_text(
            "@<TRIPOS>ATOM\n"
            "1 N         -25.7360   -7.0810   59.0550 N         1 AP1     "
            "-0.516300\n"
            "2 H         -25.2620   -6.6490   59.6280 H         1 AP1      "
            "0.333361\n"
            "3 CA        -25.9480   -6.4070   57.7830 DU        1 AP1      "
            "0.038100\n"
        )

    with (
        pytest.warns(UserWarning, match="Atom type DU found in file"),
        patch("meze.ligand.os.system", side_effect=fake_system) as mock_sys
    ):
        result = ligand._run_antechamber(
            parameterisation_directory=str(tmp_path),
            input_file=str(tmp_path / "MOL.pdb"),
            output_file=mol2_path,
        )
    assert mock_sys.call_args[0][0] == (
        f"antechamber -fi pdb -fo mol2 -i {tmp_path}/MOL.pdb "
        f"-o {mol2_path} -c bcc -nc -1 -at gaff2 -pf y -rn MOL"
    )
    assert result == mol2_path


def test_run_antechamber_missing_output_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ligand = Ligand(
        file=str(DATA / "ligands/ligand_11.pdb"), charge=-1, name="ligand_11"
    )
    mol2_path = str(tmp_path / "ligand_11.mol2")

    with patch("meze.ligand.os.system"), pytest.raises(
        RuntimeError, match="antechamber failed: missing output files"
    ):
        ligand._run_antechamber(
            parameterisation_directory=str(tmp_path),
            input_file=str(tmp_path / "MOL.pdb"),
            output_file=mol2_path,
        )


def test_parameterise_too_many_files_raises():
    with pytest.raises(ValueError, match="Expected one ligand file"):
        Ligand(
            file=[
                str(DATA / "ligands/ligand_11_solv.inpcrd"),
                str(DATA / "ligands/ligand_11_solv.prmtop")
            ],
            charge=-1,
            name="ligand_11"
        ).parameterise()


def test_parameterise_wrong_method_raises():
    with pytest.raises(ValueError, match="'method' should be one of"):
        Ligand(
            file=str(DATA / "ligands/ligand_11.pdb"),
            charge=-1,
            name="ligand_11"
        ).parameterise(method="wrong")


def test_parameterise_antechamber_method(tmp_path):
    # _run_antechamber/_run_parmchk2 are mocked out directly rather than via
    # os.system, since they're already covered on their own; the mocked
    # antechamber return value points at a real, valid file (the original
    # ligand pdb) so the real, unmocked bss.IO.readMolecules(...) call at
    # the end of parameterise() has something genuine to read.
    ligand = Ligand(
        file=str(DATA / "ligands/ligand_11.pdb"), charge=-1, name="ligand_11"
    )
    real_pdb = str(DATA / "ligands/ligand_11.pdb")
    frcmod_path = str(tmp_path / "ligand_11.frcmod")

    with patch.object(
        ligand, "_run_antechamber", return_value=real_pdb
    ) as mock_ac, patch.object(
        ligand, "_run_parmchk2", return_value=frcmod_path
    ) as mock_pc:
        result = ligand.parameterise(
            directory=str(tmp_path), method="antechamber", residue_name="MOL"
        )

    mol2_path = str(tmp_path / "ligand_11.mol2")
    assert mock_ac.call_args.kwargs["output_file"] == mol2_path
    assert mock_pc.call_args.kwargs["input_file"] == mol2_path
    assert mock_pc.call_args.kwargs["output_file"] == frcmod_path
    assert result.parameterised is True
    assert result.file == mol2_path
    assert result.frcmod_file == frcmod_path
    assert result.residue_name == "MOL"
    assert result.system.nMolecules() == 1

    # the residue-renaming logic runs for real before any mocked call
    renamed = (tmp_path / "MOL.pdb").read_text()
    assert " MOL " in [
        line for line in renamed.splitlines() if "HETATM" in line
    ][0]


def test_parameterise_tleap_method(tmp_path):
    ligand = Ligand(
        file=str(DATA / "ligands/ligand_11.pdb"), charge=-1, name="ligand_11"
    )
    real_pdb = str(DATA / "ligands/ligand_11.pdb")

    with patch.object(
        ligand, "run_ligand_tleap", return_value=real_pdb
    ) as mock_tleap:
        result = ligand.parameterise(
            directory=str(tmp_path),
            method="tleap",
            residue_name="MOL",
            force_field="gaff2",
        )

    assert mock_tleap.call_args.kwargs["coordinate_file"] == str(
        tmp_path / "MOL.pdb"
    )
    assert mock_tleap.call_args.kwargs["force_field"] == "gaff2"
    assert result.parameterised is True
    assert result.file == real_pdb
    assert result.frcmod_file is None
    assert result.residue_name == "MOL"
