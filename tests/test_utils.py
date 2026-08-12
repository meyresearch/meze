from types import SimpleNamespace
from unittest.mock import patch
import pytest
import warnings
from meze.utils import (
    _check_log_files,
    _edit_mcpbpy_tleap_input,
    _get_mol2_charge,
    _list_rindex,
    _parse_mcpbpy_input,
    _pretty,
    _remove_gpu_from_fep_configs,
    _residue_restraint_mask,
    _set_n_somd_moves,
    _write_distance_restraints,
    _write_gaussian_script,
    _write_somd_restraints,
    _write_tleap_solvation_input,
    pdb_to_sdf,
)
from unittest.mock import patch

def test_list_rindex_returns_last_matching_index():
    assert _list_rindex(["a", "bx", "c", "bx", "d"], "b") == 3


def test_list_rindex_raises_when_word_not_found():
    with pytest.raises(ValueError):
        _list_rindex(["a", "b", "c"], "z")


def test_list_rindex_match_only_at_last_position():
    assert _list_rindex(["a", "b", "c"], "c") == 2


@pytest.mark.parametrize(
    "residue_ids,expected",
    [
        ([5], "5"),
        ([1, 2], "1-2"),
        ([1, 5], "1,5"),
        ([1, 2, 3], "1-3"),
        ([1, 3, 5], "1,3,5"),
        ([1, 2, 3, 5, 7, 8, 9], "1-3,5,7-9"),
        ([3, 1, 2, 2], "1-3")
    ],
)
def test_residue_restraint_mask(residue_ids, expected):
    assert _residue_restraint_mask(residue_ids) == expected


def test_write_distance_restraints_exact_line():
    restraints = {(10, 20): (2.0, 100.0, 1.0)}
    lines = _write_distance_restraints(restraints)
    assert lines == [
        "&rst iat=10,20, r1=1.0, r2=1.5, "
        "r3=2.5, r4=3.0, rk2=100.0, rk3=100.0, /\n"
    ]


def test_write_somd_restraints_round_trip(tmp_path):
    original = {(10, 20): (2.0, 100.0, 1.0)}
    lines = _write_distance_restraints(original)

    restraint_file = tmp_path / "restraints.RST"
    restraint_file.write_text("".join(lines))

    parsed = _write_somd_restraints(str(restraint_file))

    # SOMD is 0-indexed, Amber restraint files are 1-indexed
    assert (9, 19) in parsed
    equilibrium_distance, force_constant, flat_bottom_radius = parsed[(9, 19)]
    assert equilibrium_distance == pytest.approx(2.0)
    assert force_constant == pytest.approx(100.0)
    # AMBER and SOMD restraints are defined differently
    # So the flat bottom radius will be output by the SOMD writer as
    # 0.5x of the AMBER one
    assert flat_bottom_radius == pytest.approx(0.5)


def test_write_somd_restraints_missing_file_raises():
    with pytest.raises(
        FileNotFoundError, match="Could not find AMBER-style restraint file"
    ):
        _write_somd_restraints("/nonexistent/restraints.RST")


def test_remove_gpu_from_fep_configs(tmp_path):
    cfg_file = tmp_path / "somd.cfg"
    cfg_file.write_text("ncycles = 100\ngpu = 0\nnmoves = 25000\nGPU = 1\n")

    _remove_gpu_from_fep_configs([str(cfg_file)])

    remaining = cfg_file.read_text().splitlines()
    assert "gpu = 0" not in remaining
    assert "ncycles = 100" in remaining
    assert "nmoves = 25000" in remaining


def _fake_ligand(
        name="MOL",
        frcmod_file="MOL.frcmod",
        file=("MOL.mol2",),
        residue_name=None
):
    return SimpleNamespace(
        name=name,
        frcmod_file=frcmod_file,
        file=list(file),
        residue_name=residue_name or name,
    )


def test_write_tleap_solvation_input_basic():
    lines = _write_tleap_solvation_input(
        protein_file="protein.pdb",
        ligand=_fake_ligand(),
    )
    joined = "".join(lines)
    assert "source leaprc.protein.ff14SB\n" in lines
    assert "source leaprc.water.tip3p\n" in lines
    assert "loadamberparams frcmod.ions1lm_126_tip3p\n" in lines
    assert "lig = loadmol2 MOL.mol2\n" in lines
    assert "protein = loadpdb protein.pdb\n" in lines
    assert "complex = combine {protein lig}\n" in lines
    assert any(
        line.startswith("solvateOct") and "iso" in line for line in lines
    )
    assert "addions2 complex Na+ 0\n" in lines
    assert "addions2 complex Cl- 0\n" in lines
    assert (
        "saveamberparm complex "
        "MOL_complex_solv.prmtop MOL_complex_solv.inpcrd\n"
    ) in lines
    assert joined.strip().endswith("quit")


def test_write_tleap_solvation_input_non_octahedral_box():
    lines = _write_tleap_solvation_input(
        protein_file="protein.pdb",
        ligand=_fake_ligand(),
        box_shape="cubic",
    )
    solvate_lines = [line for line in lines if line.startswith("solvate")]
    assert len(solvate_lines) == 1
    assert "iso" not in solvate_lines[0]
    assert "solvateBox" in solvate_lines[0]


def test_write_tleap_solvation_input_disulfide_bridges():
    lines = _write_tleap_solvation_input(
        protein_file="protein.pdb",
        ligand=_fake_ligand(),
        disulfide_bridges=[{"resid1": 12, "resid2": 34}],
    )
    assert "bond protein.12.SG protein.34.SG\n" in lines


def test_write_tleap_solvation_input_non_standard_residues():
    lines = _write_tleap_solvation_input(
        protein_file="protein.pdb",
        ligand=_fake_ligand(),
        non_standard_residues=[
            _fake_ligand(
                name="MOH", frcmod_file="MOH.frcmod", file=("MOH.mol2",)
            )
        ],
    )
    assert "loadamberparams MOH.frcmod\n" in lines
    assert "MOH = loadmol2 MOH.mol2\n" in lines


def test_write_tleap_input_wrong_box_shape_raises():
    with pytest.raises(ValueError):
        _write_tleap_solvation_input(
            protein_file="protein.pdb",
            ligand=_fake_ligand(),
            box_shape="orthorhombic dodecahedron"
        )


def test_edit_mcpbpy_tleap_input_substitutes_solvate_line_cubic(tmp_path):
    tleap_file = tmp_path / "mcpbpy_tleap.in"
    tleap_file.write_text(
        "mol = loadpdb complex.pdb\n"
        "solvateOct mol TIP3PBOX 10.0 iso 0.75\n"
        "quit\n"
    )

    new_lines = _edit_mcpbpy_tleap_input(str(tleap_file), box_shape="cubic")

    solvate_lines = [line for line in new_lines if line.startswith("solvate")]
    assert len(solvate_lines) == 1
    assert solvate_lines[0] == "solvateBox mol TIP3PBOX 10.0 0.75\n"
    assert "source leaprc.gaff2\n" in new_lines


def test_edit_mcpbpy_tleap_input_substitutes_solvate_line_octahedral(tmp_path):
    tleap_file = tmp_path / "mcpbpy_tleap.in"
    tleap_file.write_text(
        "mol = loadpdb complex.pdb\n"
        "solvateBox mol TIP3PBOX 10.0 0.75\n"
        "quit\n"
    )

    new_lines = _edit_mcpbpy_tleap_input(
        str(tleap_file), box_shape="octahedral"
    )

    solvate_lines = [line for line in new_lines if line.startswith("solvate")]
    assert len(solvate_lines) == 1
    assert solvate_lines[0] == "solvateOct mol TIP3PBOX 10.0 iso 0.75\n"
    assert "source leaprc.gaff2\n" in new_lines


def test_edit_mcpbpy_tleap_input_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        _edit_mcpbpy_tleap_input("/nonexistent/tleap.in")


def test_edit_mcpbpy_tleap_input_wrong_box_shape_raises(tmp_path):
    tleap_file = tmp_path / "mcpbpy_tleap.in"
    tleap_file.write_text(
        "mol = loadpdb complex.pdb\n"
        "solvateOct mol TIP3PBOX 10.0 iso 0.75\n"
        "quit\n"
    )
    with pytest.raises(ValueError):
        _edit_mcpbpy_tleap_input(
            str(tleap_file), box_shape="orthorhombic dodecahedron"
        )


def test_write_gaussian_script_content(tmp_path):
    script_path = _write_gaussian_script(
        job_name="ligand-mk",
        gaussian_version="g16",
        script_name="slurm_mk.sh",
        com_file="ligand_large_mk.com",
        directory=str(tmp_path),
        sbatch_options={"--time": "01:00:00"},
        additional_lines=["module load gaussian"],
    )

    assert script_path == str(tmp_path / "slurm_mk.sh")
    content = tmp_path.joinpath("slurm_mk.sh").read_text()
    assert content.startswith("#!/bin/bash\n")
    assert "#SBATCH --job-name=ligand-mk\n" in content
    assert "#SBATCH --time=01:00:00\n" in content
    assert "module load gaussian\n" in content
    assert content.endswith("g16 ligand_large_mk.com")


def test_parse_mcpbpy_input_type_coercion(tmp_path):
    mcpb_file = tmp_path / "mcpbpy.in"
    mcpb_file.write_text(
        "cut_off 2.8\n"
        "ion_ids 5\n"
        "gaff 2\n"
        "large_opt 1\n"
        "ion_mol2files ZN1.mol2 ZN2.mol2\n"
    )

    parsed = _parse_mcpbpy_input(str(mcpb_file))

    assert parsed["cut_off"] == pytest.approx(2.8)
    assert parsed["ion_ids"] == 5
    assert parsed["gaff"] == 2
    assert parsed["large_opt"] == 1
    assert parsed["ion_mol2files"] == ["ZN1.mol2", "ZN2.mol2"]


def test_parse_mcpbpy_input_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        _parse_mcpbpy_input("/nonexistent/mcpbpy.in")


def test_parse_mcpbpy_input_empty_path_raises():
    with pytest.raises(ValueError):
        _parse_mcpbpy_input(mcpbpy_input_file=None)


def test_check_log_files_warns_when_none_found(tmp_path, caplog):
    result = _check_log_files(str(tmp_path))
    assert result == []
    assert "Could not find any log files" in caplog.text


def test_check_log_files_warns_on_single_log_file(tmp_path, caplog):
    log_file = tmp_path / "ligand.log"
    log_file.write_text("Normal termination of Gaussian\n")
    _check_log_files(str(tmp_path))
    assert "Only one log file found" in caplog.text


def test_check_log_files_warns_on_abnormal_termination(tmp_path, caplog):
    (tmp_path / "a.log").write_text("Normal termination of Gaussian\n")
    (tmp_path / "b.log").write_text("Error termination\n")
    _check_log_files(str(tmp_path))
    assert "did not terminate normally" in caplog.text


def test_check_log_files_empty_log_raises_ioerror(tmp_path):
    (tmp_path / "a.log").write_text("")
    (tmp_path / "b.log").write_text("Normal termination of Gaussian\n")
    with pytest.raises(IOError):
        _check_log_files(str(tmp_path))


def test_check_log_file_all_not_converged(tmp_path, caplog):
    (tmp_path / "large_opt.log").write_text(
        "         Item               Value     Threshold  Converged?\n"
        "Maximum Force            0.225680     0.000450     NO\n"
        "RMS     Force            0.049275     0.000300     NO\n"
        "Maximum Displacement     0.051861     0.001800     NO\n"
        "RMS     Displacement     0.010123     0.001200     NO\n"
    )
    _check_log_files(str(tmp_path))
    assert "did not converge" in caplog.text


def test_check_log_file_some_not_converged(tmp_path, caplog):
    (tmp_path / "large_opt.log").write_text(
        "         Item               Value     Threshold  Converged?\n"
        "Maximum Force            0.225680     0.000450     NO\n"
        "RMS     Force            0.049275     0.000300     NO\n"
        "Maximum Displacement     0.051861     0.001800     YES\n"
        "RMS     Displacement     0.010123     0.001200     YES\n"
    )
    _check_log_files(str(tmp_path))
    assert "did not converge" in caplog.text


def test_check_log_file_converged(tmp_path):
    (tmp_path / "large_opt.log").write_text(
        "         Item               Value     Threshold  Converged?\n"
        "Maximum Force            0.225680     0.000450     NO\n"
        "RMS     Force            0.049275     0.000300     NO\n"
        "Maximum Displacement     0.051861     0.001800     NO\n"
        "RMS     Displacement     0.010123     0.001200     NO\n"
        "         Item               Value     Threshold  Converged?\n"
        "Maximum Force            0.225680     0.000450     YES\n"
        "RMS     Force            0.049275     0.000300     YES\n"
        "Maximum Displacement     0.051861     0.001800     YES\n"
        "RMS     Displacement     0.010123     0.001200     YES\n"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _check_log_files(str(tmp_path))
    assert not any(
        "did not converge" in str(warning.message) for warning in caught
    )


def test_get_mol2_charge_sums_atom_charges(tmp_path):
    mol2_file = tmp_path / "MOL.mol2"
    mol2_file.write_text(
        "@<TRIPOS>MOLECULE\n"
        "MOL\n"
        "@<TRIPOS>ATOM\n"
        "      1 C1    0.0000   0.0000   0.0000 C.3    1  MOL      0.100000\n"
        "      2 C2    1.0000   0.0000   0.0000 C.3    1  MOL     -0.050000\n"
        "@<TRIPOS>BOND\n"
        "      1    1    2   1\n"
    )
    assert _get_mol2_charge(str(mol2_file)) == pytest.approx(0.05)


def test_get_mol2_charge_malformed_file_raises(tmp_path):
    mol2_file = tmp_path / "malformed.mol2"
    mol2_file.write_text("not a real mol2 file\n")
    with pytest.raises(RuntimeError):
        _get_mol2_charge(str(mol2_file))


def test_pdb_to_sdf_single_string_input(tmp_path):
    pdb_file = tmp_path / "ligand_1.pdb"
    pdb_file.write_text("dummy pdb content")

    with patch("meze.utils.os.system") as mock_system:
        output_files = pdb_to_sdf(str(pdb_file))

    expected_output = str(tmp_path / "ligand_1.sdf")
    assert output_files == [expected_output]
    command = mock_system.call_args[0][0]
    assert command == f"obabel -i pdb {pdb_file} -o sdf -O {expected_output}"


def test_pdb_to_sdf_list_input(tmp_path):
    pdb_files = [tmp_path / "a.pdb", tmp_path / "b.pdb"]
    for f in pdb_files:
        f.write_text("dummy")

    with patch("meze.utils.os.system"):
        output_files = pdb_to_sdf([str(f) for f in pdb_files])

    assert output_files == [str(tmp_path / "a.sdf"), str(tmp_path / "b.sdf")]


def test_pdb_to_sdf_empty_input_raises():
    with pytest.raises(RuntimeError):
        pdb_to_sdf([])


@pytest.mark.parametrize(
    "sampling_time,n_somd_cycles,stepsize,expected",
    [
        (1.0, 10, 0.002, 50_000),
        (4.0, 20, 0.002, 100_000),
        (1.0, 4, 0.001, 250_000),
    ],
)
def test_set_n_somd_moves(sampling_time, n_somd_cycles, stepsize, expected):
    assert _set_n_somd_moves(
        sampling_time, n_somd_cycles, stepsize
    ) == expected


def test_set_n_somd_moves_default_stepsize():
    assert _set_n_somd_moves(1.0, 10) == 50_000


def test_pretty_dataclass():
    from dataclasses import dataclass

    @dataclass
    class Foo:
        a: int
        b: str

    assert _pretty(Foo(1, "x")) == "Foo(\n  a=1,\n  b='x'\n)"


def test_pretty_dict():
    assert _pretty({"x": 1, "y": 2}) == "{\n  'x': 1,\n  'y': 2\n}"


def test_pretty_single_item_list():
    assert _pretty([1]) == "[1]"


def test_pretty_multi_item_list():
    assert _pretty([1, 2]) == "[\n  1,\n  2\n]"
