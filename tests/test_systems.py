import os
import tempfile
import pytest
from meze import (
    ColdMeze, ColdMezeRecipe, Ligand, MezeRecipe, Meze,
    HotMeze, HotMezeRecipe, ColdQuantumMeze, HotQuantumMeze,
    QuantumMeze
)
import shutil
from pathlib import Path
from unittest.mock import patch
import MDAnalysis as mda
import dataclasses

DATA = Path(__file__).parent.parent / "tests" / "data"


def test_meze_save_no_suffix(vim2_cold_meze, tmp_path):
    dummy_file = str(tmp_path / "dummy_file")
    saved_file = vim2_cold_meze.save(dummy_file)
    assert os.path.isfile(saved_file)
    assert Path(saved_file).suffix == ".pkl"


def test_meze_save_with_suffix(vim2_cold_meze, tmp_path):
    dummy_file = str(tmp_path / "dummy_file.pkl")
    saved_file = vim2_cold_meze.save(dummy_file)
    assert os.path.isfile(saved_file)
    assert Path(saved_file).suffix == ".pkl"


def test_meze_load_no_file_raises():
    with pytest.raises(FileNotFoundError, match="Pickle meze file not found"):
        Meze.load(filename="nonexistent/file")


def test_meze_save_load_round_trip(vim2_cold_meze, tmp_path):
    saved_file = vim2_cold_meze.save(str(tmp_path / "meze"))
    loaded = Meze.load(saved_file)
    assert loaded.topology == vim2_cold_meze.topology
    assert loaded.coordinates == vim2_cold_meze.coordinates
    assert loaded.recipe == vim2_cold_meze.recipe
    assert loaded.stage == vim2_cold_meze.stage
    assert loaded.non_standard_residues == vim2_cold_meze.non_standard_residues
    assert loaded.ligand.name == vim2_cold_meze.ligand.name
    assert loaded.ligand.charge == vim2_cold_meze.ligand.charge
    assert loaded.ligand.residue_name == vim2_cold_meze.ligand.residue_name


def test_cold_meze_wrong_recipe_type_raises():
    with pytest.raises(
        TypeError, match="Expected 'recipe' to be a ColdMezeRecipe"
    ):
        ColdMeze.from_files(
            pdb_file=str(DATA / "vim2/vim2.fixed.pdb"), recipe=123
        )


def test_hot_meze_wrong_recipe_type_raises():
    with pytest.raises(
        TypeError, match="Expected 'recipe' to be a HotMezeRecipe"
    ):
        HotMeze.from_files(
            pdb_file=str(DATA / "vim2/vim2.fixed.pdb"), recipe=123
        )


def test_cold_quantum_meze_wrong_recipe_type_raises():
    with pytest.raises(
        TypeError, match="Expected 'recipe' to be a ColdMezeRecipe"
    ):
        ColdQuantumMeze.from_files(
            topology=str(DATA / "vim2/vim2_complex.prmtop"),
            coordinates=str(DATA / "vim2/vim2_complex.inpcrd"),
            recipe=123
        )


def test_hot_quantum_meze_wrong_recipe_type_raises():
    with pytest.raises(
        TypeError, match="Expected 'recipe' to be a HotMezeRecipe"
    ):
        HotQuantumMeze.from_files(
            topology=str(DATA / "vim2/vim2_complex.prmtop"),
            coordinates=str(DATA / "vim2/vim2_complex.inpcrd"),
            recipe=123
        )


def test_cold_meze_no_files_raises():
    with pytest.raises(ValueError, match="must supply either a pdb file"):
        ColdMeze.from_files(recipe=ColdMezeRecipe())


def test_hot_meze_no_files_raises():
    with pytest.raises(ValueError, match="must supply either a pdb file"):
        HotMeze.from_files(recipe=HotMezeRecipe())


def test_cold_meze_recipe_none_built_from_kwargs():
    m = ColdMeze.from_files(
        pdb_file=str(DATA / "vim2/vim2.fixed.pdb"),
        metal="ZN",
        group_name="from_kwargs"
    )
    assert isinstance(m.recipe, ColdMezeRecipe)
    assert m.recipe.metal == "Zn"
    assert m.recipe.group_name == "from_kwargs"


def test_cold_meze_recipe_dict_built_from_dict():
    m = ColdMeze.from_files(
        pdb_file=str(DATA / "vim2/vim2.fixed.pdb"),
        recipe={"metal": "ZN", "group_name": "from_dict"}
    )
    assert isinstance(m.recipe, ColdMezeRecipe)
    assert m.recipe.metal == "Zn"
    assert m.recipe.group_name == "from_dict"


def test_hot_meze_recipe_none_built_from_kwargs():
    m = HotMeze.from_files(
        pdb_file=str(DATA / "vim2/vim2.fixed.pdb"),
        metal="ZN",
        group_name="from_kwargs"
    )
    assert isinstance(m.recipe, HotMezeRecipe)
    assert m.recipe.metal == "Zn"
    assert m.recipe.group_name == "from_kwargs"


def test_hot_meze_recipe_dict_built_from_dict():
    m = HotMeze.from_files(
        pdb_file=str(DATA / "vim2/vim2.fixed.pdb"),
        recipe={"metal": "ZN", "group_name": "from_dict"}
    )
    assert isinstance(m.recipe, HotMezeRecipe)
    assert m.recipe.metal == "Zn"
    assert m.recipe.group_name == "from_dict"


@pytest.mark.parametrize("resid,expected", [
    (83, "1247-1257"),
    (85, "1284-1294"),
    (87, "1313-1318"),
    (148, "2183-2193"),
    (167, "2455-2458"),
    (209, "3088-3098"),
])
def test_get_side_chain_selection(resid, expected):
    qm_meze = QuantumMeze.from_files(
        topology=str(DATA / "vim2/vim2_complex.prmtop"),
        coordinates=str(DATA / "vim2/vim2_complex.inpcrd"),
        recipe=MezeRecipe()
    )
    residue = qm_meze.universe.select_atoms(f"resid {resid}").residues[0]
    assert qm_meze._get_side_chain_selection(residue) == expected


def test_cold_quantum_meze_recipe_none_built_from_kwargs():
    qm = ColdQuantumMeze.from_files(
        topology=str(DATA / "vim2/vim2_complex.prmtop"),
        coordinates=str(DATA / "vim2/vim2_complex.inpcrd"),
        metal="ZN",
        group_name="from_kwargs"
    )
    assert isinstance(qm.recipe, ColdMezeRecipe)
    assert qm.recipe.metal == "Zn"
    assert qm.recipe.group_name == "from_kwargs"


def test_cold_quantum_meze_recipe_dict_built_from_dict():
    qm = ColdQuantumMeze.from_files(
        topology=str(DATA / "vim2/vim2_complex.prmtop"),
        coordinates=str(DATA / "vim2/vim2_complex.inpcrd"),
        recipe={"metal": "ZN", "group_name": "from_dict"}
    )
    assert isinstance(qm.recipe, ColdMezeRecipe)
    assert qm.recipe.metal == "Zn"
    assert qm.recipe.group_name == "from_dict"


def test_hot_quantum_meze_recipe_none_built_from_kwargs():
    hqm = HotQuantumMeze.from_files(
        topology=str(DATA / "vim2/vim2_complex.prmtop"),
        coordinates=str(DATA / "vim2/vim2_complex.inpcrd"),
        metal="ZN",
        group_name="from_kwargs"
    )
    assert isinstance(hqm.recipe, HotMezeRecipe)
    assert hqm.recipe.metal == "Zn"
    assert hqm.recipe.group_name == "from_kwargs"


def test_hot_quantum_meze_recipe_dict_built_from_dict():
    hqm = HotQuantumMeze.from_files(
        topology=str(DATA / "vim2/vim2_complex.prmtop"),
        coordinates=str(DATA / "vim2/vim2_complex.inpcrd"),
        recipe={"metal": "ZN", "group_name": "from_dict"}
    )
    assert isinstance(hqm.recipe, HotMezeRecipe)
    assert hqm.recipe.metal == "Zn"
    assert hqm.recipe.group_name == "from_dict"


def test_write_qm_namelist():
    qm_meze = QuantumMeze.from_files(
        topology=str(DATA / "vim2/vim2_complex.prmtop"),
        coordinates=str(DATA / "vim2/vim2_complex.inpcrd"),
        recipe=MezeRecipe()
    )
    qm_namelist = qm_meze._write_qm_namelist()
    assert qm_namelist[0] == "&qmmm"
    assert qm_namelist[-1] == "/"
    assert "  writepdb=1" in qm_namelist
    assert "  qmcharge=0" in qm_namelist
    assert "  qm_theory='DFTB3'" in qm_namelist
    assert "  qmshake=0" in qm_namelist
    assert "  qm_ewald=1" in qm_namelist
    assert "  qm_pme=1" in qm_namelist

    qmmask_line = next(line for line in qm_namelist if "qmmask=" in line)
    assert qmmask_line.startswith("  qmmask=':235-236|(@")
    assert qmmask_line.endswith(")'")
    atom_ids = qmmask_line.split("(@")[1].rstrip(")'").split(",")
    assert set(atom_ids) == {
        "3088-3098", "3450", "2455-2458", "1313-1318",
        "1284-1294", "2183-2193", "3451", "1247-1257"
    }


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


def test_set_universe_no_elements_pdb(vim2_recipe_json):
    no_elements = ColdMeze.from_files(
        pdb_file=str(DATA / "vim2/vim2_no_elements.pdb"),
        recipe=ColdMezeRecipe(**vim2_recipe_json)
    )
    elements = no_elements.universe.atoms.elements
    assert len(elements) == len(no_elements.universe.atoms)
    assert set(elements) >= {"H", "C", "N", "O", "S", "ZN"}
    assert "" not in elements


def test_meze_with_topology_and_coordinates(vim2_top_and_coord):
    meze = vim2_top_and_coord
    assert len(meze.universe.atoms) == 25009
    assert list(meze.metal_resids) == [233, 234]
    assert list(meze.metal_atomids) == [3450, 3451]
    assert set(meze.coordinating_residues.keys()) == {3450, 3451}


def test_base_meze_with_topology_and_coordinates():
    meze = Meze(
        topology=str(DATA / "vim2/vim2_complex.prmtop"),
        coordinates=str(DATA / "vim2/vim2_complex.inpcrd"),
        recipe=MezeRecipe()
    )
    assert len(meze.universe.atoms) == 25009
    assert list(meze.metal_resids) == [233, 234]
    assert list(meze.metal_atomids) == [3450, 3451]
    assert set(meze.coordinating_residues.keys()) == {3450, 3451}
    assert set(meze.crystal_waters.resnames) == {"WAT"}
    assert meze.exclude_resids == []


def test_ligand_resid_set_from_parameterised_ligand():
    ligand = Ligand(
        file=str(DATA / "ligands/ligand_11.pdb"),
        name="ligand_11",
        charge=-1,
        parameterised=True,
        residue_name="MOL",
    )
    meze = Meze(
        topology=str(DATA / "vim2/vim2_complex.prmtop"),
        coordinates=str(DATA / "vim2/vim2_complex.inpcrd"),
        recipe=MezeRecipe(),
        ligand=ligand,
    )
    assert meze.ligand_resid == 236
    assert meze.ligand_resname == "MOL"


def test_ligand_resid_preset_skips_recompute():
    ligand = Ligand(
        file=str(DATA / "ligands/ligand_11.pdb"),
        name="ligand_11",
        charge=-1,
        parameterised=True,
        residue_name="MOL",
    )
    meze = Meze(
        topology=str(DATA / "vim2/vim2_complex.prmtop"),
        coordinates=str(DATA / "vim2/vim2_complex.inpcrd"),
        recipe=MezeRecipe(),
        ligand=ligand,
        ligand_resid=236,
    )
    assert meze.ligand_resid == 236
    assert meze.ligand_resname == "MOL"


def test_set_ligand_infers_from_ligand_resname():
    meze = Meze(
        topology=str(DATA / "vim2/vim2_complex.prmtop"),
        coordinates=str(DATA / "vim2/vim2_complex.inpcrd"),
        recipe=MezeRecipe(),
        ligand_resname="MOL",
    )
    assert meze.ligand is not None
    assert meze.ligand.residue_name == "MOL"
    assert meze.ligand.parameterised is True
    assert meze.ligand.file == [
        str(DATA / "vim2/vim2_complex.inpcrd"),
        str(DATA / "vim2/vim2_complex.prmtop"),
    ]
    assert meze.ligand.charge == pytest.approx(-0.999, abs=1e-2)
    assert meze.ligand_resid == 236


def test_set_metal_raises_for_absent_metal(vim2_recipe_json):
    bad_recipe = ColdMezeRecipe(**{**vim2_recipe_json, "metal": "FE"})
    with pytest.raises(ValueError, match="No atoms found for metal"):
        ColdMeze.from_files(
            recipe=bad_recipe,
            pdb_file=str(DATA / "vim2/vim2.fixed.pdb"),
        )


def test_cold_meze_empty_exclude_resids(vim2_cold_meze):
    assert vim2_cold_meze.exclude_resids == set()


def test_cold_meze_exclude_resids(vim2_recipe_json):
    m = ColdMeze.from_files(
        recipe=ColdMezeRecipe(**vim2_recipe_json),
        pdb_file=str(DATA / "vim2/vim2.fixed.pdb"),
        exclude_resids=5
    )
    assert m.exclude_resids == {5}


def test_metal_and_coordinating_residues(vim2_cold_meze):
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


def test_select_crystal_waters(vim2_cold_meze):
    assert set(vim2_cold_meze.crystal_waters.resnames) == {"WAT"}
    assert vim2_cold_meze.crystal_waters.n_residues == 182


def test_build_distance_restraints(vim2_cold_meze):
    restraints = vim2_cold_meze.build_distance_restraints(

    )
    assert len(restraints) == 8
    assert restraints[(3450, 1255)] == (2.05, 100.0, 1.0)
    assert restraints[(3450, 1288)] == (2.07, 100.0, 1.0)
    assert restraints[(3450, 2191)] == (2.05, 100.0, 1.0)
    assert restraints[(3450, 3452)] == (1.89, 100.0, 1.0)
    assert restraints[(3451, 3452)] == (2.08, 100.0, 1.0)
    assert restraints[(3451, 1318)] == (2.35, 100.0, 1.0)
    assert restraints[(3451, 2458)] == (2.29, 100.0, 1.0)
    assert restraints[(3451, 3096)] == (2.15, 100.0, 1.0)


def test_build_distance_restraints_wrong_type_raises(vim2_cold_meze):
    with pytest.raises(
        TypeError, match="coordinating_residues must be a dict"
    ):
        vim2_cold_meze.build_distance_restraints(
            coordinating_residues="not a dict"
        )


def test_build_distance_restraints_wrong_value_type_raises(vim2_cold_meze):
    with pytest.raises(
        TypeError, match="coordinating_residues values must be "
    ):
        vim2_cold_meze.build_distance_restraints(
            coordinating_residues={3450: "some string"}
        )


def test_build_distance_restraints_wrong_key_type_raises(vim2_cold_meze):
    with pytest.raises(
        TypeError, match="coordinating_residues keys must be "
    ):
        vim2_cold_meze.build_distance_restraints(
            coordinating_residues={"string": "some string"}
        )


def test_ligand_not_in_distance_restraints(vim2_top_and_coord):
    restraints = vim2_top_and_coord.build_distance_restraints()
    for (_, ligand_id) in restraints.keys():
        assert ligand_id != 3473


def test_ligand_not_in_angle_restraints(vim2_top_and_coord):
    restraints = vim2_top_and_coord.build_angle_restraints()
    for (_, ligand_id) in restraints.keys():
        assert ligand_id != 3473


def test_build_angle_restraints(vim2_cold_meze):
    restraints = vim2_cold_meze.build_angle_restraints()
    assert len(restraints) == 12
    assert restraints[(1255, 1288)] == (3.08, 100.0, 1.0)
    assert restraints[(1255, 2191)] == (3.22, 100.0, 1.0)
    assert restraints[(1255, 3452)] == (3.27, 100.0, 1.0)
    assert restraints[(1288, 2191)] == (3.34, 100.0, 1.0)
    assert restraints[(1288, 3452)] == (3.34, 100.0, 1.0)
    assert restraints[(2191, 3452)] == (3.38, 100.0, 1.0)
    assert restraints[(1318, 2458)] == (3.55, 100.0, 1.0)
    assert restraints[(1318, 3096)] == (2.96, 100.0, 1.0)
    assert restraints[(1318, 3452)] == (2.87, 100.0, 1.0)
    assert restraints[(2458, 3096)] == (3.48, 100.0, 1.0)
    assert restraints[(2458, 3452)] == (3.7, 100.0, 1.0)
    assert restraints[(3096, 3452)] == (3.97, 100.0, 1.0)


def test_build_custom_distance_restraints_mismatched_lengths_raises(
        vim2_cold_meze
):
    with pytest.raises(ValueError, match="force_constant has"):
        vim2_cold_meze.build_custom_distance_restraints(
            atom_pairs=[("resid 1 and name CA", "resid 2 and name CA")],
            force_constant=[100.0, 200.0],
        )


def test_build_custom_distance_restraints_bad_selection_raises(
        vim2_cold_meze
):
    with pytest.raises(ValueError, match="must match exactly one atom"):
        vim2_cold_meze.build_custom_distance_restraints(
            atom_pairs=[("resid 1", "resid 2 and name CA")],
        )


def test_build_restraint_mask_solute(vim2_cold_meze):
    mask = vim2_cold_meze._build_restraint_mask(position_restraints="solute")
    assert mask == "':1-235'"


def test_build_restraint_mask_backbone(vim2_cold_meze):
    mask = vim2_cold_meze._build_restraint_mask(position_restraints="backbone")
    assert mask == "'(@N,CA,C,O & !:WAT)|:83,85,87,148,167,209,233-235'"


def test_build_restraint_mask_metal_coordination(vim2_cold_meze):
    mask = vim2_cold_meze._build_restraint_mask(
        position_restraints="metal-coordination"
    )
    assert mask == "':83,85,87,148,167,209,233-235'"


def test_build_restraint_mask_none(vim2_cold_meze):
    assert vim2_cold_meze._build_restraint_mask(
        position_restraints=None
    ) is None


def test_build_restraint_mask_invalid_option_raises(vim2_cold_meze):
    with pytest.raises(ValueError, match="Invalid restraint option"):
        vim2_cold_meze._build_restraint_mask(position_restraints="bogus")


def test_additional_restraints_dict_raises(vim2_cold_meze):
    with pytest.raises(ValueError, match="additional_restraints must contain"):
        vim2_cold_meze._build_restraint_mask(
            position_restraints="solute",
            additional_restraints={"wrong": "values"}
        )


def test_build_restraint_mask_additional(vim2_cold_meze):
    mask = vim2_cold_meze._build_restraint_mask(
        position_restraints=None,
        additional_restraints={"resids": [236], "resnames": ["MOH"]}
    )
    assert mask == "':235-236'"


def test_validate_disulfide_bridges_self_bonded_raises(vim2_cold_meze):
    vim2_cold_meze.disulfide_bridges = [{"resid1": 5, "resid2": 5}]
    with pytest.raises(ValueError, match="cannot connect residue 5 to itself"):
        vim2_cold_meze._validate_disulfide_bridges()


def test_validate_disulfide_bridges_missing_keys_raises(vim2_cold_meze):
    vim2_cold_meze.disulfide_bridges = [{"resid1": 1}]
    with pytest.raises(ValueError, match="Invalid disulfide bridge entry"):
        vim2_cold_meze._validate_disulfide_bridges()


def test_validate_disulfide_bridges_non_cyx_raises(vim2_cold_meze):
    vim2_cold_meze.disulfide_bridges = [{"resid1": 1, "resid2": 2}]
    with pytest.raises(ValueError, match="require CYX residues"):
        vim2_cold_meze._validate_disulfide_bridges()


def test_validate_disulfide_bridges_no_conect(l1_recipe_json):
    l1_meze = ColdMeze.from_files(
        pdb_file=str(DATA / "l1/l1_no_conect.pdb"),
        recipe=ColdMezeRecipe(**l1_recipe_json),
        disulfide_bridges=[{"resid1": 217, "resid2": 245}]
    )
    l1_meze._validate_disulfide_bridges()
    assert l1_meze.disulfide_bridges == [{"resid1": 217, "resid2": 245}]


def test_validate_disulfide_bridges_with_conect(l1_recipe_json, caplog):
    l1_meze = ColdMeze.from_files(
        pdb_file=str(DATA / "l1/l1_with_conect.pdb"),
        recipe=ColdMezeRecipe(**l1_recipe_json),
        disulfide_bridges=[{"resid1": 217, "resid2": 245}]
    )
    l1_meze._validate_disulfide_bridges()
    assert (
        "Residues 217 and 245 appear to already have a disulfide bond"
    ) in caplog.text
    assert l1_meze.disulfide_bridges is None


def test_validate_duplicate_disulfide_bridges(l1_recipe_json):
    l1_meze = ColdMeze.from_files(
        pdb_file=str(DATA / "l1/l1_no_conect.pdb"),
        recipe=ColdMezeRecipe(**l1_recipe_json),
        disulfide_bridges=[{"resid1": 217, "resid2": 245},
                           {"resid1": 217, "resid2": 245}]
    )
    with pytest.raises(ValueError, match="Duplicate disulfide bridge"):
        l1_meze._validate_disulfide_bridges()


def test_validate_missing_disulfide_bridges(l1_recipe_json):
    l1_meze = ColdMeze.from_files(
        pdb_file=str(DATA / "l1/l1_no_conect.pdb"),
        recipe=ColdMezeRecipe(**l1_recipe_json),
        disulfide_bridges=[{"resid1": 217, "resid2": 245},
                           {"resid1": 1089, "resid2": 1514}]
    )
    with pytest.raises(
        ValueError, match="Residue 1089 or 1514 not found in structure"
    ):
        l1_meze._validate_disulfide_bridges()


def test_validate_no_cyx_disulfide_bridges(l1_recipe_json):
    l1_meze = ColdMeze.from_files(
        pdb_file=str(DATA / "l1/l1_no_SG.pdb"),
        recipe=ColdMezeRecipe(**l1_recipe_json),
        disulfide_bridges=[{"resid1": 217, "resid2": 245}]
    )
    with pytest.raises(
        ValueError, match="Missing SG atom"
    ):
        l1_meze._validate_disulfide_bridges()


def test_validate_long_disulfide_bridges(l1_recipe_json):
    l1_meze = ColdMeze.from_files(
        pdb_file=str(DATA / "l1/l1_long_ssbond.pdb"),
        recipe=ColdMezeRecipe(**l1_recipe_json),
        disulfide_bridges=[{"resid1": 217, "resid2": 245}]
    )
    with pytest.raises(
        ValueError, match="Disulfide 217-245 too long:"
    ):
        l1_meze._validate_disulfide_bridges()


def test_write_restrained_atoms_pdb(vim2_cold_meze):
    with open(DATA / "vim2/test_restraints.pdb", "r") as f:
        reference = f.readlines()
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "restrained.pdb")
        vim2_cold_meze.write_restrained_atoms_pdb(out)
        assert os.path.isfile(out)
        with open(out) as f:
            content = f.readlines()

    assert reference == content


def test_get_mutatable_ligand_no_matching_residue_raises(vim2_top_and_coord):
    vim2_top_and_coord.ligand_resname = "ZZZ"
    with pytest.raises(
        RuntimeError, match="Could not find any ligand residues"
    ):
        vim2_top_and_coord.get_mutatable_ligand_molecule()


def test_get_mutatable_ligand_multiple_residues_raises(vim2_top_and_coord):
    vim2_top_and_coord.ligand_resname = "WAT"
    with pytest.raises(NotImplementedError, match="multiple residues"):
        vim2_top_and_coord.get_mutatable_ligand_molecule()


def test_get_mutatable_ligand_molecule(vim2_top_and_coord):
    vim2_top_and_coord.ligand_resname = "MOL"
    molecule = vim2_top_and_coord.get_mutatable_ligand_molecule()
    assert molecule.nAtoms() == 31


def test_get_mutatable_ligand_no_resname_raises(vim2_recipe_json):
    meze = ColdMeze.from_files(
        topology=str(DATA / "vim2/vim2_complex.prmtop"),
        coordinates=str(DATA / "vim2/vim2_complex.inpcrd"),
        recipe=ColdMezeRecipe(**vim2_recipe_json)
    )
    with pytest.raises(RuntimeError, match="Ligand residue name not set"):
        meze.get_mutatable_ligand_molecule()


def test_set_protein(vim2_cold_meze):
    assert vim2_cold_meze.protein.n_residues == 232


def test_set_protein_raises():
    with pytest.raises(RuntimeError, match="Could not set protein"):
        Meze(
            topology=str(DATA / "vim2/water.pdb"),
            coordinates=str(DATA / "vim2/water.pdb"),
            recipe=MezeRecipe(),
            stage="bound"
        )


def test_unbound_stage_skips_protein_check():
    meze = Meze(
        topology=str(DATA / "vim2/water.pdb"),
        coordinates=str(DATA / "vim2/water.pdb"),
        recipe=MezeRecipe(),
        stage="unbound"
    )
    assert meze.protein is None


def test_unknown_stage_raises():
    with pytest.raises(ValueError, match="Unrecognised stage"):
        Meze(
            topology=str(DATA / "vim2/water.pdb"),
            coordinates=str(DATA / "vim2/water.pdb"),
            recipe=MezeRecipe(),
            stage="something"
        )


def test_hot_meze_happy_path(vim2_recipe_json):
    hm = HotMeze.from_files(
        pdb_file=str(DATA / "vim2/vim2.fixed.pdb"),
        recipe=HotMezeRecipe(**vim2_recipe_json)
    )
    assert hm.restraint_file is None


def test_hot_meze_restraint_file_missing_raises(vim2_recipe_json):
    with pytest.raises(FileNotFoundError, match="Restraint file not found"):
        HotMeze.from_files(
            pdb_file=str(DATA / "vim2/vim2.fixed.pdb"),
            recipe=HotMezeRecipe(**vim2_recipe_json),
            restraint_file="nonexistent.RST"
        )


def test_hot_meze_restraint_file_real_ok(vim2_recipe_json, tmp_path):
    restraint_file = tmp_path / "restraints.RST"
    restraint_file.write_text("dummy")
    hm = HotMeze.from_files(
        pdb_file=str(DATA / "vim2/vim2.fixed.pdb"),
        recipe=HotMezeRecipe(**vim2_recipe_json),
        restraint_file=str(restraint_file)
    )
    assert hm.restraint_file == str(restraint_file)


def test_hot_meze_model_zero_warns_no_restraint_file(vim2_recipe_json, caplog):
    HotMeze.from_files(
        pdb_file=str(DATA / "vim2/vim2.fixed.pdb"),
        recipe=HotMezeRecipe(**{**vim2_recipe_json, "model": 0})
    )
    assert "No restraint file supplied while model is 0" in caplog.text


def test_cold_quantum_meze_happy_path(vim2_recipe_json):
    qm = ColdQuantumMeze.from_files(
        topology=str(DATA / "vim2/vim2_complex.prmtop"),
        coordinates=str(DATA / "vim2/vim2_complex.inpcrd"),
        recipe=ColdMezeRecipe(**vim2_recipe_json)
    )
    assert set(qm.qm_region["whole_residues"]) == {235, 236}
    assert set(qm.qm_region["atom_ids"]) == {
        "1313-1318", "1284-1294", "2455-2458", "1247-1257",
        "3450", "3088-3098", "2183-2193", "3451"
    }
    assert qm.qm_charge == 0
    assert qm.distance_restraints is None


def test_quantum_meze_exclude_resids_int_normalized(vim2_recipe_json):
    qm = ColdQuantumMeze.from_files(
        topology=str(DATA / "vim2/vim2_complex.prmtop"),
        coordinates=str(DATA / "vim2/vim2_complex.inpcrd"),
        recipe=ColdMezeRecipe(**vim2_recipe_json),
        exclude_resids=234
    )
    assert qm.exclude_resids == {234}
    assert qm.qm_region["whole_residues"] == []
    assert set(qm.qm_region["atom_ids"]) == {
        "1284-1294", "1247-1257", "3450", "2183-2193"
    }


def test_quantum_meze_metal_resids_for_distance_restraints_int_normalized(
    vim2_recipe_json
):
    qm = ColdQuantumMeze.from_files(
        topology=str(DATA / "vim2/vim2_complex.prmtop"),
        coordinates=str(DATA / "vim2/vim2_complex.inpcrd"),
        recipe=ColdMezeRecipe(**vim2_recipe_json),
        metal_resids_for_distance_restraints=234
    )
    assert qm.metal_resids_for_distance_restraints == [234]


def test_quantum_meze_additional_qm_resids_int_normalized(vim2_recipe_json):
    qm = ColdQuantumMeze.from_files(
        topology=str(DATA / "vim2/vim2_complex.prmtop"),
        coordinates=str(DATA / "vim2/vim2_complex.inpcrd"),
        recipe=ColdMezeRecipe(**vim2_recipe_json),
        additional_qm_resids=1
    )
    assert qm._additional_qm_resids == {1}
    assert "2-15" in qm.qm_region["atom_ids"]


def test_quantum_meze_additional_qm_resnames_resolved(vim2_recipe_json):
    qm = ColdQuantumMeze.from_files(
        topology=str(DATA / "vim2/vim2_complex.prmtop"),
        coordinates=str(DATA / "vim2/vim2_complex.inpcrd"),
        recipe=ColdMezeRecipe(**vim2_recipe_json),
        additional_qm_resnames="MOH"
    )
    assert qm._additional_qm_resids == {235}


def test_quantum_meze_custom_qm_region_not_dict_raises(vim2_recipe_json):
    with pytest.raises(TypeError, match="custom_qm_region must be a dict"):
        ColdQuantumMeze.from_files(
            topology=str(DATA / "vim2/vim2_complex.prmtop"),
            coordinates=str(DATA / "vim2/vim2_complex.inpcrd"),
            recipe=ColdMezeRecipe(**vim2_recipe_json),
            custom_qm_region="not a dict"
        )


def test_quantum_meze_custom_qm_region_missing_keys_raises(vim2_recipe_json):
    with pytest.raises(ValueError, match="missing required keys"):
        ColdQuantumMeze.from_files(
            topology=str(DATA / "vim2/vim2_complex.prmtop"),
            coordinates=str(DATA / "vim2/vim2_complex.inpcrd"),
            recipe=ColdMezeRecipe(**vim2_recipe_json),
            custom_qm_region={"whole_residues": [1]}
        )


def test_quantum_meze_custom_qm_region_bad_whole_residues_type_raises(
    vim2_recipe_json
):
    with pytest.raises(
        TypeError, match="whole_residues'\\] must be an int or list of int"
    ):
        ColdQuantumMeze.from_files(
            topology=str(DATA / "vim2/vim2_complex.prmtop"),
            coordinates=str(DATA / "vim2/vim2_complex.inpcrd"),
            recipe=ColdMezeRecipe(**vim2_recipe_json),
            custom_qm_region={"whole_residues": "x", "atom_ids": []}
        )


def test_quantum_meze_custom_qm_region_bad_atom_ids_type_raises(
        vim2_recipe_json
):
    with pytest.raises(TypeError, match="atom_ids'\\] must be a list of str"):
        ColdQuantumMeze.from_files(
            topology=str(DATA / "vim2/vim2_complex.prmtop"),
            coordinates=str(DATA / "vim2/vim2_complex.inpcrd"),
            recipe=ColdMezeRecipe(**vim2_recipe_json),
            custom_qm_region={"whole_residues": [1], "atom_ids": [1, 2]}
        )


def test_quantum_meze_custom_qm_region_used_directly(vim2_recipe_json):
    custom = {"whole_residues": [5], "atom_ids": ["10-12"]}
    qm = ColdQuantumMeze.from_files(
        topology=str(DATA / "vim2/vim2_complex.prmtop"),
        coordinates=str(DATA / "vim2/vim2_complex.inpcrd"),
        recipe=ColdMezeRecipe(**vim2_recipe_json),
        custom_qm_region=custom
    )
    assert qm.qm_region == custom
    assert qm.qm_charge == 0


def test_hot_quantum_meze_happy_path(vim2_recipe_json):
    hqm = HotQuantumMeze.from_files(
        topology=str(DATA / "vim2/vim2_complex.prmtop"),
        coordinates=str(DATA / "vim2/vim2_complex.inpcrd"),
        recipe=HotMezeRecipe(**vim2_recipe_json)
    )
    assert hqm.qm_charge == 0


def test_prepare_metals_for_ezaff(vim2_cold_meze, tmp_path):
    def fake_system(cmd):
        out_path = cmd.split("-o ")[1].split(" -c")[0]
        Path(out_path).write_text("dummy mol2\n")

    with patch(
        "meze.sofra.os.system", side_effect=fake_system
    ) as mock_sys:
        result = vim2_cold_meze.prepare_metals_for_ezaff(
            directory=str(tmp_path)
        )

    assert result == ["ZN1.mol2", "ZN2.mol2"]
    assert mock_sys.call_count == 2
    assert mock_sys.call_args_list[0][0][0] == (
        f"metalpdb2mol2.py -i {tmp_path}/ZN1.pdb "
        f"-o {tmp_path}/ZN1.mol2 -c 2"
    )
    assert mock_sys.call_args_list[1][0][0] == (
        f"metalpdb2mol2.py -i {tmp_path}/ZN2.pdb "
        f"-o {tmp_path}/ZN2.mol2 -c 2"
    )
    zn1 = mda.Universe(str(tmp_path / "ZN1.pdb"))
    assert len(zn1.atoms) == 1
    zn2 = mda.Universe(str(tmp_path / "ZN2.pdb"))
    assert len(zn2.atoms) == 1


def test_prepare_metals_for_ezaff_raises_missing_output_error(
        vim2_cold_meze, tmp_path
):
    with patch("meze.sofra.os.system"), pytest.raises(
        RuntimeError, match="Could not prepare"
    ):
        vim2_cold_meze.prepare_metals_for_ezaff(directory=str(tmp_path))


def test_write_complex(vim2_cold_meze, tmp_path):
    shutil.copy(
        DATA / "ligands/ligand_11.pdb",
        tmp_path / f"{vim2_cold_meze.ligand.name}.pdb"
    )

    with patch("meze.sofra.os.system") as mock_sys:
        result = vim2_cold_meze.write_complex(
            directory=str(tmp_path), ligand_name="ligand_11"
        )

    assert mock_sys.call_args[0][0] == (
        f"pdb4amber -i {tmp_path}/ligand_11_complex.pdb "
        f"-o {tmp_path}/vim2_model_0_ligand_11.amber.pdb"
    )
    assert result.filename == str(
        tmp_path / "vim2_model_0_ligand_11.amber.pdb"
    )
    assert len(result.atoms) == 4029

    complex_pdb = tmp_path / "ligand_11_complex.pdb"
    assert complex_pdb.is_file()
    written = mda.Universe(str(complex_pdb))
    assert len(written.atoms) == 4029


def test_build_empirical_bonds(vim2_cold_meze, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cwd_before = os.getcwd()
    meze = dataclasses.replace(
        vim2_cold_meze,
        mcpbpy_input_file="some/mcpbpy.in",
        parameterisation_directory=str(tmp_path)
    )

    with patch("meze.sofra.os.system") as mock_sys:
        result = meze.build_empirical_bonds()

    assert mock_sys.call_args[0][0] == (
        f"MCPB.py -i some/mcpbpy.in -s 2e > {tmp_path}/mcpb_step2e.out"
    )
    assert result.mcpbpy_input_file == "some/mcpbpy.in"
    assert os.getcwd() == cwd_before


def test_prepare_resp_calculation_wrong_additional_lines_type_raises(
        vim2_cold_meze
):
    with pytest.raises(
        TypeError, match="additional_lines must be a list of strings"
    ):
        vim2_cold_meze.prepare_resp_calculation(
            additional_lines=[
                "string",
                "string2",
                1234
            ]
        )


def test_prepare_resp_calculation_paramdirectory_not_set(vim2_cold_meze):
    meze = dataclasses.replace(vim2_cold_meze, parameterisation_directory=None)
    with pytest.raises(
        ValueError, match="MCPB parameterisation directory not set"
    ):
        meze.prepare_resp_calculation()


def test_prepare_resp_calculation_no_com_files(
        vim2_cold_meze, tmp_path
):
    meze = dataclasses.replace(
        vim2_cold_meze, parameterisation_directory=str(tmp_path)
    )
    with pytest.raises(RuntimeError, match="No Gaussian .com files found in "):
        meze.prepare_resp_calculation()


def test_prepare_resp_calculation(vim2_cold_meze, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cwd_before = os.getcwd()
    meze = dataclasses.replace(
        vim2_cold_meze,
        parameterisation_directory=str(tmp_path)
    )

    com_content = (
        "%Mem=8000MB\n"
        "%NProcShared=8\n"
        "%Chk=ligand_large_mk.chk\n"
        "# B3LYP/6-31G* Opt Pop=MK\n"
        "\n"
        "comment\n"
        "\n"
        "CLR\n"
        "0 1\n"
        "Zn 0.0 0.0 0.0\n"
        "Zn\n"
    )

    def fake_system(cmd):
        if "MCPB.py" in cmd:
            (tmp_path / "ligand_large_mk.com").write_text(com_content)

    with patch(
        "meze.sofra.os.system", side_effect=fake_system
    ) as mock_sys:
        result = meze.prepare_resp_calculation()

    mcpbpy_input_file = str(tmp_path / "mcpbpy.in")
    assert mock_sys.call_args_list[0][0][0] == (
        f"MCPB.py -i {mcpbpy_input_file} -s 1 > {tmp_path}/mcpb_step1.out"
    )
    assert mock_sys.call_args_list[1][0][0] == (
        f"chmod +x {tmp_path}/ligand_slurm_g_opt.sh"
    )
    assert mock_sys.call_args_list[2][0][0] == (
        f"chmod +x {tmp_path}/ligand_slurm_mk.sh"
    )
    assert result.mcpbpy_input_file == mcpbpy_input_file
    assert (tmp_path / "ligand_large_opt.com").is_file()
    assert (tmp_path / "ligand_slurm_g_opt.sh").is_file()
    assert (tmp_path / "ligand_slurm_mk.sh").is_file()
    assert os.getcwd() == cwd_before


def test_add_water_raises_wrong_non_standard_param_method(vim2_cold_meze):
    with pytest.raises(
        TypeError, match="non_standard_parameterisation_method must be"
    ):
        vim2_cold_meze.add_water(
            non_standard_parameterisation_method="wrong"
        )


def test_add_water_raises_wrong_model_option(vim2_cold_meze):
    meze = ColdQuantumMeze.from_files(
        topology=str(DATA / "vim2/vim2_complex.prmtop"),
        coordinates=str(DATA / "vim2/vim2_complex.inpcrd"),
        model=5
    )
    with pytest.raises(
        NotImplementedError, match="Model option 5 is not implemented"
    ):
        meze.add_water()


def test_add_water_model_2_builds_tleap_command(vim2_cold_meze, tmp_path):
    meze = dataclasses.replace(
        vim2_cold_meze,
        recipe=vim2_cold_meze.recipe.model_copy(update={"model": 2}),
        non_standard_residues=None
    )

    with patch("meze.sofra.os.system") as mock_sys, pytest.raises(
        RuntimeError, match="Failed to solvate meze"
    ):
        meze.add_water(directory=str(tmp_path))

    assert mock_sys.call_args[0][0] == (
        f"tleap -s -f {tmp_path}/tleap_solvate.in > "
        f"{tmp_path}/tleap_solvate.out"
    )
    assert (tmp_path / "tleap_solvate.in").is_file()
