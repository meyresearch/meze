import os
import dataclasses
from unittest.mock import patch
import BioSimSpace as bss
import pytest


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
        vim2_top_and_coord, tmp_path, mock_amber_process
):
    recipe = vim2_top_and_coord.recipe.model_copy(
        update={"workdir": str(tmp_path), "model": 2}
    )
    meze = dataclasses.replace(vim2_top_and_coord, recipe=recipe)
    protocol = bss.Protocol.Minimisation(steps=100)

    with pytest.raises(
        RuntimeError, match="The run test-run excited with an error"
    ), patch(
        "meze.sofra.bss.Process.Amber", side_effect=mock_amber_process
    ):
        meze._run(
            system=None,
            recipe=recipe,
            protocol=protocol,
            process_name="test-run",
            config_options={},
            is_gpu=False
        )
