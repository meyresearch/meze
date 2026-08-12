import os
from meze import Sofra
import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch


def _fake_os_system(cmd):
    out_path = cmd.split(" -O ")[1]
    Path(out_path).write_text("dummy sdf\n")


def _build_sofra(vim2_cold_meze, tmp_path):
    pickle_file = vim2_cold_meze.save(str(tmp_path / "meze.pkl"))
    sofra_file = str(tmp_path / "sofra.json")
    vim2_cold_meze.add_to_sofra(
        filename=sofra_file, key="ligand_11", pickle_file=pickle_file
    )
    return Sofra.from_file(sofra_file=sofra_file)


def test_sofra_from_file_without_network_file(vim2_cold_meze, tmp_path):
    sofra_file = str(tmp_path / "sofra.json")
    sofra = _build_sofra()

    assert list(sofra.mezes.keys()) == ["ligand_11"]
    assert isinstance(sofra.mezes["ligand_11"], type(vim2_cold_meze))
    assert sofra.network_file is None
    assert sofra.group_name == "vim2_model_0"
    assert sofra.project_directory == os.getcwd()
    assert sofra.sofra_file == sofra_file


def test_sofra_from_file_with_network_file(tmp_path):
    sofra_file = str(tmp_path / "sofra.json")
    sofra = _build_sofra()

    network_file = str(tmp_path / "network.csv")
    with open(network_file, "w") as f:
        f.write("Name_1,Name_2,Score\n")
    sofra.save_network_file(network_file)

    reloaded = Sofra.from_file(sofra_file=sofra_file)

    assert reloaded.network_file == network_file
    assert list(reloaded.mezes.keys()) == ["ligand_11"]


def test_no_sofra_file_raises():
    with pytest.raises(FileNotFoundError, match="Sofra file not found: "):
        Sofra.from_file(sofra_file="nonexistent/sofra")


def test_no_network_file_raises(tmp_path):
    sofra_file = tmp_path / "sofra.json"
    sofra_file.write_text('{"network_file": "nonexistent/file"}')
    with pytest.raises(FileNotFoundError, match="Could not find network file"):
        Sofra.from_file(sofra_file=str(sofra_file))


def test_entry_missing_pickle_file_skipped(vim2_cold_meze, tmp_path, caplog):
    pickle_file = vim2_cold_meze.save(str(tmp_path / "meze.pkl"))
    sofra_file = tmp_path / "sofra.json"
    sofra_file.write_text(json.dumps({
        "ligand_11": {"pickle_file": pickle_file},
        "ligand_12": {"not_pickle_file": "oops"}
    }))

    sofra = Sofra.from_file(sofra_file=str(sofra_file))

    assert list(sofra.mezes.keys()) == ["ligand_11"]
    assert "Could not find pickle file for ligand_12" in caplog.text


def test_no_mezes_raises(tmp_path):
    sofra_file = tmp_path / "sofra.json"
    sofra_file.write_text("{}")
    with pytest.raises(RuntimeError, match="No pickle mezes found"):
        Sofra.from_file(sofra_file=str(sofra_file))


def test_only_one_meze_warning(vim2_cold_meze, tmp_path, caplog):
    pickle_file = vim2_cold_meze.save(str(tmp_path / "meze.pkl"))
    sofra_file = tmp_path / "sofra.json"
    sofra_file.write_text(json.dumps({
        "ligand_11": {"pickle_file": pickle_file},
    }))

    Sofra.from_file(sofra_file=str(sofra_file))
    assert "Found only one meze in" in caplog.text


def test_sofra_from_file_wrong_directory(vim2_cold_meze, tmp_path):
    pickle_file = vim2_cold_meze.save(str(tmp_path / "meze.pkl"))
    sofra_file = str(tmp_path / "sofra.json")
    vim2_cold_meze.add_to_sofra(
        filename=sofra_file, key="ligand_11", pickle_file=pickle_file
    )

    with pytest.raises(
        FileNotFoundError, match="Project directory nonexistent/ does not exist"
    ):
        Sofra.from_file(sofra_file=sofra_file, directory="nonexistent/")


def test_sofra_parse_lomap_output(tmp_path):
    sofra = Sofra(
        mezes={},
        sofra_file="dummy.json",
        sofra_contents={"a": "b"}
    )
    lomap_file = tmp_path / "lomap.txt"
    lomap_file.write_text(
        "Index_1   ,Index_2   ,Filename_1               ,Filename_2           "
        "    ,Str_sim        ,Eff_sim        ,Loose_sim      ,Connect   \n"
        "0         ,1         ,ligand_11.sdf            ,ligand_12.sdf        "
        "    ,0.74082        ,0.74082        ,0.74082        ,Yes       ,"
        "0:19,1:0,2:1,3:2,4:3,5:4,6:5,7:6,8:7,9:8,10:9,11:10,12:11,13:12,14:13"
        ",15:14,16:15,17:16,18:17,19:18,20:25,21:26,22:24,23:27,24:28,25:29,"
        "26:30,27:31,28:21,29:32,30:33"
    )
    transf, scores, file = sofra._parse_lomap_output(
        lomap_file,
        tmp_path
    )
    assert transf == [("ligand_11", "ligand_12")]
    assert scores == [0.74082]
    assert file == str(tmp_path / f"{sofra.group_name}_lomap_network.csv")
    assert (tmp_path / file).read_text() == (
        "Name_1,Name_2,Score\n"
        "ligand_11,ligand_12,0.74082\n"
    )


def test_sofra_parse_lomap_output_raises(tmp_path):
    sofra = Sofra(
        mezes={},
        sofra_file="dummy.json",
        sofra_contents={"a": "b"}
    )
    lomap_file = tmp_path / "lomap.txt"
    lomap_file.write_text(
        "Index_1   ,Index_2   ,Filename_1               ,Filename_2           "
        "    ,Str_sim        ,Eff_sim        ,Loose_sim      ,Connect   \n"
        "0         ,1         ,ligand_11.sdf            ,ligand_12.sdf        "
        "    ,0.74082        ,0.74082        ,0.74082        ,No       ,"
        "0:19,1:0,2:1,3:2,4:3,5:4,6:5,7:6,8:7,9:8,10:9,11:10,12:11,13:12,14:13"
        ",15:14,16:15,17:16,18:17,19:18,20:25,21:26,22:24,23:27,24:28,25:29,"
        "26:30,27:31,28:21,29:32,30:33"
    )
    with pytest.raises(RuntimeError, match="Lomap output did not contain any "):
        sofra._parse_lomap_output(
            lomap_file,
            tmp_path
        )


def test_sofra_parse_lomap_output_mixed_rows(tmp_path):
    sofra = Sofra(
        mezes={},
        sofra_file="dummy.json",
        sofra_contents={"a": "b"}
    )
    lomap_file = tmp_path / "lomap.txt"
    lomap_file.write_text(
        "Index_1   ,Index_2   ,Filename_1               ,Filename_2           "
        "    ,Str_sim        ,Eff_sim        ,Loose_sim      ,Connect   \n"
        "0         ,1         ,ligand_11.sdf            ,ligand_12.sdf        "
        "    ,0.74082        ,0.74082        ,0.74082        ,Yes       ,"
        "0:19,1:0,2:1,3:2,4:3,5:4,6:5,7:6,8:7,9:8,10:9,11:10,12:11,13:12,14:13"
        ",15:14,16:15,17:16,18:17,19:18,20:25,21:26,22:24,23:27,24:28,25:29,"
        "26:30,27:31,28:21,29:32,30:33"
        "2         ,3         ,ligand_11.sdf            ,ligand_12.sdf        "
        "    ,0.74082        ,0.74082        ,0.74082        ,No       ,"
        "0:19,1:0,2:1,3:2,4:3,5:4,6:5,7:6,8:7,9:8,10:9,11:10,12:11,13:12,14:13"
        ",15:14,16:15,17:16,18:17,19:18,20:25,21:26,22:24,23:27,24:28,25:29,"
        "26:30,27:31,28:21,29:32,30:33"
    )
    transf, scores, file = sofra._parse_lomap_output(
        lomap_file,
        tmp_path
    )
    assert transf == [("ligand_11", "ligand_12")]
    assert scores == [0.74082]
    assert file == str(tmp_path / f"{sofra.group_name}_lomap_network.csv")
    assert (tmp_path / file).read_text() == (
        "Name_1,Name_2,Score\n"
        "ligand_11,ligand_12,0.74082\n"
    )


def test_set_ligand_network_happy_path(vim2_cold_meze, tmp_path):
    sofra = _build_sofra(vim2_cold_meze, tmp_path)
    pdb_files = [str(tmp_path / "a.pdb"), str(tmp_path / "b.pdb")]
    for f in pdb_files:
        Path(f).write_text("dummy pdb\n")

    def fake_subprocess_run(cmd, **kwargs):
        lomap_dir = os.path.join(str(tmp_path), "lomap")
        scores_file = os.path.join(
            lomap_dir, f"{sofra.group_name}_score_with_connection.txt"
        )
        with open(scores_file, "w") as f:
            f.write("Filename_1,Filename_2,Str_sim,Connect\n")
            f.write("a.sdf,b.sdf,0.5,Yes\n")
        Path(
            os.path.join(lomap_dir, f"{sofra.group_name}.png")
        ).write_text("x")
        result = Mock()
        result.returncode = 0
        result.stderr = ""
        return result

    workdir_before = os.getcwd()
    with patch("meze.utils.os.system", side_effect=_fake_os_system), \
         patch("meze.sofra.subprocess.run", side_effect=fake_subprocess_run):
        sofra.set_ligand_network(pdb_files=pdb_files, directory=str(tmp_path))

    assert sofra.transformations == [("a", "b")]
    assert sofra.lomap_scores == [0.5]
    assert sofra.network_file.endswith("_lomap_network.csv")
    assert os.path.isfile(sofra.network_file)
    assert os.getcwd() == workdir_before


def test_set_ligand_network_raises_when_scores_file_missing(
        vim2_cold_meze, tmp_path
):
    sofra = _build_sofra(vim2_cold_meze, tmp_path)
    pdb_files = [str(tmp_path / "a.pdb")]
    Path(pdb_files[0]).write_text("dummy pdb\n")

    workdir_before = os.getcwd()
    mock_result = Mock(returncode=0, stderr="")
    with patch("meze.utils.os.system", side_effect=_fake_os_system), \
         patch("meze.sofra.subprocess.run", return_value=mock_result), \
         pytest.raises(RuntimeError, match="Lomap did not produce"):
        sofra.set_ligand_network(pdb_files=pdb_files, directory=str(tmp_path))

    assert os.getcwd() == workdir_before
