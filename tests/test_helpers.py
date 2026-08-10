from unittest.mock import patch
import pytest
from meze.helpers import _check_ambertools


def test_check_ambertools_all_found_on_path():
    with patch("meze.helpers.shutil.which", return_value="/usr/bin/tool"):
        _check_ambertools()


def test_check_ambertools_found_via_amberhome():
    with patch("meze.helpers.shutil.which", return_value=None), \
         patch.dict("os.environ", {"AMBERHOME": "/opt/amber"}, clear=True), \
         patch("meze.helpers.os.path.exists", return_value=True):
        _check_ambertools()


def test_check_ambertools_missing_raises():
    with patch("meze.helpers.shutil.which", return_value=None), \
         patch.dict("os.environ", {}, clear=True):
        with pytest.raises(
            RuntimeError, match="AmberTools installation required"
        ):
            _check_ambertools()
