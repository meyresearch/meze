import os
import dataclasses
from pathlib import Path
from unittest.mock import patch
import BioSimSpace as bss
import pytest
from meze.utils import _write_distance_restraints


def test_run_happy_path(vim2_top_and_coord, tmp_path, mock_amber_process):
    recipe = vim2_top_and_coord.recipe.model_copy(
        update={"workdir": str(tmp_path), "model": 2}
    )
    meze = dataclasses.replace(vim2_top_and_coord, recipe=recipe)
    protocol = bss.Protocol.Minimisation(steps=100)

    with patch(
        "meze.sofra.bss.Process.Amber", side_effect=mock_amber_process
    ) as mock_amber:
        result = meze._run(
            system=None,
            recipe=recipe,
            protocol=protocol,
            process_name="test-run",
            config_options={"cut": 12.0},
            is_gpu=False
        )

    call_kwargs = mock_amber.call_args.kwargs
    run_directory = str(tmp_path / "test-run")
    assert call_kwargs["system"] is meze.system
    assert call_kwargs["protocol"] is protocol
    assert call_kwargs["work_dir"] == run_directory
    assert call_kwargs["name"] == "test-run"
    assert call_kwargs["extra_options"] == {"cut": 12.0}
    assert call_kwargs["extra_lines"] == []
    assert call_kwargs["is_gpu"] is False
    assert call_kwargs["exe"] == recipe.path_to_engine

    process = mock_amber_process.processes[-1]
    process.start.assert_called_once()
    process.wait.assert_called_once()

    assert result.topology == os.path.join(run_directory, "next.prm7")
    assert result.coordinates == os.path.join(run_directory, "next.rst7")
    assert os.path.isfile(result.topology)
    assert os.path.isfile(result.coordinates)
    assert result.recipe == recipe


def test_run_process_is_error_raises(
        vim2_top_and_coord, tmp_path, mock_amber_process_error
):
    recipe = vim2_top_and_coord.recipe.model_copy(
        update={"workdir": str(tmp_path), "model": 2}
    )
    meze = dataclasses.replace(vim2_top_and_coord, recipe=recipe)
    protocol = bss.Protocol.Minimisation(steps=100)

    with pytest.raises(
        RuntimeError, match="The run test-run exited with an error"
    ), patch(
        "meze.sofra.bss.Process.Amber", side_effect=mock_amber_process_error
    ):
        meze._run(
            system=None,
            recipe=recipe,
            protocol=protocol,
            process_name="test-run",
            config_options={},
            is_gpu=False
        )


def test_run_distance_restraints_branch(
        vim2_top_and_coord, tmp_path, mock_amber_process
):
    recipe = vim2_top_and_coord.recipe.model_copy(
        update={"workdir": str(tmp_path), "model": 0}
    )
    meze = dataclasses.replace(vim2_top_and_coord, recipe=recipe)
    protocol = bss.Protocol.Minimisation(steps=100)
    distance_restraints = _write_distance_restraints(
        {(3450, 1255): (2.05, 100.0, 1.0)}
    )

    with patch("meze.sofra.bss.Process.Amber", side_effect=mock_amber_process):
        meze._run(
            system=None,
            recipe=recipe,
            protocol=protocol,
            process_name="test-run",
            config_options={},
            is_gpu=False,
            distance_restraints=distance_restraints
        )

    run_directory = tmp_path / "test-run"
    assert (run_directory / "test-run_restrained_atoms.pdb").is_file()

    coordination_restraints = meze._prepare_distance_restraints()
    angle_restraints = meze._prepare_angle_restraints()
    expected_restraints = (
        distance_restraints + coordination_restraints + angle_restraints
    )

    top_level_restraints = tmp_path / "restraints.RST"
    assert top_level_restraints.read_text() == "".join(expected_restraints)
    assert (run_directory / "restraints.RST").read_text() == (
        top_level_restraints.read_text()
    )

    config_content = (run_directory / "test-run.cfg").read_text()
    assert "&wt TYPE='DUMPFREQ', istep1=100 /" in config_content
    assert '&wt TYPE="END", /' in config_content
    assert "DISANG=restraints.RST" in config_content
    assert "DUMPAVE=distances.out" in config_content


def test_run_restraint_file_remap_branch(
        vim2_top_and_coord, tmp_path, mock_amber_process
):
    recipe = vim2_top_and_coord.recipe.model_copy(
        update={"workdir": str(tmp_path)}
    )
    restraint_input = tmp_path / "my_restraints.RST"
    original_lines = _write_distance_restraints(
        {(3450, 1255): (2.05, 100.0, 1.0)}
    )
    restraint_input.write_text("".join(original_lines))

    meze = dataclasses.replace(
        vim2_top_and_coord,
        recipe=recipe,
        restraint_file=str(restraint_input)
    )
    protocol = bss.Protocol.Minimisation(steps=100)

    with patch("meze.sofra.bss.Process.Amber", side_effect=mock_amber_process):
        meze._run(
            system=None,
            recipe=recipe,
            protocol=protocol,
            process_name="test-run",
            config_options={},
            is_gpu=False
        )

    run_directory = tmp_path / "test-run"

    assert meze.restraint_file == str(run_directory / "my_restraints.RST")
    assert os.path.isfile(meze.restraint_file)
    remapped_lines = Path(meze.restraint_file).read_text().splitlines(
        keepends=True
    )
    # Does NOT exercise this issue: github.com/OpenBioSim/sire issue #423
    assert remapped_lines == original_lines

    coordination_restraints = meze._prepare_distance_restraints()
    angle_restraints = meze._prepare_angle_restraints()
    expected_restraints = (
        remapped_lines + coordination_restraints + angle_restraints
    )

    top_level_restraints = tmp_path / "restraints.RST"
    assert top_level_restraints.read_text() == "".join(expected_restraints)
    assert (run_directory / "restraints.RST").read_text() == (
        top_level_restraints.read_text()
    )


def test_coldmeze_run_wrong_protocol_raises(vim2_cold_meze):
    with pytest.raises(ValueError, match="Unsupported protocol type 'wrong'"):
        vim2_cold_meze.run(
            protocol_type="wrong"
        )


def test_run_minimisation_protocol_type(
        vim2_top_and_coord, tmp_path, mock_amber_process
):
    recipe = vim2_top_and_coord.recipe.model_copy(
        update={"workdir": str(tmp_path), "model": 0}
    )
    meze = dataclasses.replace(vim2_top_and_coord, recipe=recipe)

    with patch(
        "meze.sofra.bss.Process.Amber", side_effect=mock_amber_process
    ) as mock_amber:
        meze.run(
            workdir=str(tmp_path),
            protocol_type="minimisation",
            process_name="test-run",
            is_gpu=False
        )

    call_kwargs = mock_amber.call_args.kwargs
    protocol = call_kwargs["protocol"]
    assert isinstance(protocol, bss.Protocol.Minimisation)
    assert protocol.getSteps() == recipe.max_cycles

    extra_options = call_kwargs["extra_options"]
    assert extra_options["nmropt"] == 1
    assert extra_options["ntmin"] == recipe.min_method
    assert extra_options["maxcyc"] == recipe.max_cycles
    assert extra_options["ncyc"] == recipe.n_sd_cycles


def test_run_nvt_ramp_up(
        vim2_top_and_coord, tmp_path, mock_amber_process
):
    recipe = vim2_top_and_coord.recipe.model_copy(
        update={"workdir": str(tmp_path), "model": 2}
    )
    meze = dataclasses.replace(vim2_top_and_coord, recipe=recipe)

    with patch(
        "meze.sofra.bss.Process.Amber", side_effect=mock_amber_process
    ) as mock_amber:
        meze.run(
            workdir=str(tmp_path),
            protocol_type="nvt",
            process_name="test-run",
            is_gpu=False,
            restart=True,
            start_temperature=100,
            end_temperature=300
        )

    call_kwargs = mock_amber.call_args.kwargs
    protocol = call_kwargs["protocol"]
    assert isinstance(protocol, bss.Protocol.Equilibration)
    assert protocol.getStartTemperature()._value == 100
    assert protocol.getEndTemperature()._value == 300
    assert protocol.getTimeStep() == recipe.dt
    assert protocol.getRunTime() == recipe.runtime
    assert protocol.getPressure() is None
    assert protocol.getForceConstant()._value == recipe.restraint_weight

    extra_options = call_kwargs["extra_options"]
    assert extra_options["irest"] == 1
    assert extra_options["ntx"] == 5


def test_run_nvt_constant_temp(
        vim2_top_and_coord, tmp_path, mock_amber_process
):
    recipe = vim2_top_and_coord.recipe.model_copy(
        update={"workdir": str(tmp_path), "model": 2}
    )
    meze = dataclasses.replace(vim2_top_and_coord, recipe=recipe)

    with patch(
        "meze.sofra.bss.Process.Amber", side_effect=mock_amber_process
    ) as mock_amber:
        meze.run(
            workdir=str(tmp_path),
            protocol_type="nvt",
            process_name="test-run",
            is_gpu=False,
            restart=True,
            temperature=300
        )

    call_kwargs = mock_amber.call_args.kwargs
    protocol = call_kwargs["protocol"]
    assert (
        protocol.getStartTemperature() == protocol.getEndTemperature()
        == recipe.temperature
    )


def test_run_npt(
        vim2_top_and_coord, tmp_path, mock_amber_process
):
    recipe = vim2_top_and_coord.recipe.model_copy(
        update={"workdir": str(tmp_path), "model": 2}
    )
    meze = dataclasses.replace(vim2_top_and_coord, recipe=recipe)

    with patch(
        "meze.sofra.bss.Process.Amber", side_effect=mock_amber_process
    ) as mock_amber:
        meze.run(
            workdir=str(tmp_path),
            protocol_type="npt",
            process_name="test-run",
            is_gpu=False,
            restart=True
        )

    call_kwargs = mock_amber.call_args.kwargs
    protocol = call_kwargs["protocol"]
    assert isinstance(protocol, bss.Protocol.Equilibration)
    assert (
        protocol.getStartTemperature() == protocol.getEndTemperature()
        == recipe.temperature
    )
    assert protocol.getPressure() == recipe.pressure
    assert call_kwargs["extra_options"]["barostat"] == recipe.barostat


