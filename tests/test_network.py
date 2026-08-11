import os
from meze import Sofra
import pytest


def test_sofra_from_file_without_network_file(vim2_cold_meze, tmp_path):
    pickle_file = vim2_cold_meze.save(str(tmp_path / "meze.pkl"))
    sofra_file = str(tmp_path / "sofra.json")
    vim2_cold_meze.add_to_sofra(
        filename=sofra_file, key="ligand_11", pickle_file=pickle_file
    )

    sofra = Sofra.from_file(sofra_file=sofra_file)

    assert list(sofra.mezes.keys()) == ["ligand_11"]
    assert isinstance(sofra.mezes["ligand_11"], type(vim2_cold_meze))
    assert sofra.network_file is None
    assert sofra.group_name == "vim2_model_0"
    assert sofra.project_directory == os.getcwd()
    assert sofra.sofra_file == sofra_file


def test_sofra_from_file_with_network_file(vim2_cold_meze, tmp_path):
    pickle_file = vim2_cold_meze.save(str(tmp_path / "meze.pkl"))
    sofra_file = str(tmp_path / "sofra.json")
    vim2_cold_meze.add_to_sofra(
        filename=sofra_file, key="ligand_11", pickle_file=pickle_file
    )
    sofra = Sofra.from_file(sofra_file=sofra_file)

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
    with pytest.raises(RuntimeError):
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
