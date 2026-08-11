import os
import tempfile
import pytest
from meze import ColdMeze, ColdMezeRecipe, Ligand, MezeRecipe, Meze
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

# # ---------------------------------------------------------------------------
# # Meze._validate_disulfide_bridges
# #
# # Note: vim2.fixed.pdb has no CYX residues at all, so branches that need a
# # *successfully validated* bridge to be reached first (duplicate-bridge
# # detection, the "already bonded in CONECT" skip, the distance-too-long
# # check) aren't reachable with this fixture -- they need a system with real
# # CYX residues to test.
# # ---------------------------------------------------------------------------

# def test_validate_disulfide_bridges_self_bonded_raises(vim2_cold_meze):
#     vim2_cold_meze.disulfide_bridges = [{"resid1": 5, "resid2": 5}]
#     with pytest.raises(ValueError, match="cannot connect residue 5 to itself"):
#         vim2_cold_meze._validate_disulfide_bridges()


# def test_validate_disulfide_bridges_missing_keys_raises(vim2_cold_meze):
#     vim2_cold_meze.disulfide_bridges = [{"resid1": 1}]
#     with pytest.raises(ValueError, match="Invalid disulfide bridge entry"):
#         vim2_cold_meze._validate_disulfide_bridges()


# def test_validate_disulfide_bridges_non_cyx_raises(vim2_cold_meze):
#     vim2_cold_meze.disulfide_bridges = [{"resid1": 1, "resid2": 2}]
#     with pytest.raises(ValueError, match="require CYX residues"):
#         vim2_cold_meze._validate_disulfide_bridges()


# ---------------------------------------------------------------------------
# Meze._validate_non_standard_residues (auto-called during construction
# when non_standard_residues is a dict)
# ---------------------------------------------------------------------------

# # ---------------------------------------------------------------------------
# # Meze.write_restrained_atoms_pdb
# # ---------------------------------------------------------------------------

# def test_write_restrained_atoms_pdb(vim2_cold_meze):
#     with tempfile.TemporaryDirectory() as d:
#         out = os.path.join(d, "restrained.pdb")
#         vim2_cold_meze.write_restrained_atoms_pdb(out)
#         assert os.path.isfile(out)
#         with open(out) as f:
#             content = f.read()
#         assert "ATOM" in content or "HETATM" in content
