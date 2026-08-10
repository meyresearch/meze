import json
import pytest
import BioSimSpace as bss
from pydantic import ValidationError
from meze import (
    MezeRecipe,
    ColdMezeRecipe,
    HotMezeRecipe,
    AlchemicalMezeRecipe,
)


def test_model_none_passthrough():
    assert MezeRecipe(model=None).model is None


def test_model_coerced_to_int():
    assert MezeRecipe(model="0").model == 0


def test_model_invalid_raises():
    with pytest.raises(ValidationError, match="a valid integer"):
        MezeRecipe(model="abc")


def test_temperature_coerced_to_bss_type():
    recipe = MezeRecipe(temperature=300.0)
    assert isinstance(recipe.temperature, bss.Types.Temperature)


def test_temperature_bss_type_passthrough():
    temperature = bss.Types.Temperature(310.0, "kelvin")
    recipe = MezeRecipe(temperature=temperature)
    assert recipe.temperature is temperature


def test_temperature_negative_raises():
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        MezeRecipe(temperature=-1.0)


def test_pressure_coerced_to_bss_type():
    recipe = MezeRecipe(pressure=1.0)
    assert isinstance(recipe.pressure, bss.Types.Pressure)


def test_pressure_negative_raises():
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        MezeRecipe(pressure=-1.0)


def test_nb_cutoff_coerced_to_bss_type():
    recipe = MezeRecipe(nb_cutoff=12.0)
    assert isinstance(recipe.nb_cutoff, bss.Types.Length)


def test_nb_cutoff_negative_raises():
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        MezeRecipe(nb_cutoff=-1.0)


def test_cold_recipe_dt_and_runtime_coerced_to_picoseconds():
    recipe = ColdMezeRecipe(dt=0.001, runtime=100.0)
    assert isinstance(recipe.dt, bss.Types.Time)
    assert isinstance(recipe.runtime, bss.Types.Time)
    assert recipe.runtime.picoseconds().value() == pytest.approx(100.0)


def test_cold_recipe_negative_runtime_raises():
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        ColdMezeRecipe(runtime=-1.0)


def test_cold_recipe_temperature_range_coerced():
    recipe = ColdMezeRecipe(start_temperature=100.0, end_temperature=300.0)
    assert isinstance(recipe.start_temperature, bss.Types.Temperature)
    assert isinstance(recipe.end_temperature, bss.Types.Temperature)


def test_cold_recipe_negative_start_temperature_raises():
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        ColdMezeRecipe(start_temperature=-1.0)


def test_hot_recipe_runtime_coerced_to_nanoseconds():
    recipe = HotMezeRecipe(runtime=100.0)
    assert isinstance(recipe.runtime, bss.Types.Time)
    assert recipe.runtime.nanoseconds().value() == pytest.approx(100.0)


def test_hot_recipe_negative_runtime_raises():
    with pytest.raises(ValueError, match="greater than 0"):
        HotMezeRecipe(runtime=0)


def test_hot_recipe_dt_coerced_to_picoseconds():
    recipe = HotMezeRecipe(dt=0.002)
    assert isinstance(recipe.dt, bss.Types.Time)
    assert recipe.dt.picoseconds().value() == pytest.approx(0.002)


def test_hot_recipe_negative_dt_raises():
    with pytest.raises(ValueError, match="greater than 0"):
        HotMezeRecipe(dt=0)


def test_cold_and_hot_runtime_use_different_units():
    cold = ColdMezeRecipe(runtime=1.0)
    hot = HotMezeRecipe(runtime=1.0)
    cold_in_ns = cold.runtime.nanoseconds().value()
    hot_in_ns = hot.runtime.nanoseconds().value()
    assert hot_in_ns == pytest.approx(cold_in_ns * 1000)


def test_alchemical_recipe_sampling_time_coerced_to_nanoseconds():
    recipe = AlchemicalMezeRecipe(sampling_time=4.0)
    assert isinstance(recipe.sampling_time, bss.Types.Time)
    assert recipe.sampling_time.nanoseconds().value() == pytest.approx(4.0)


def test_alchemical_recipe_zero_sampling_time_raises():
    with pytest.raises(ValueError, match="greater than 0"):
        AlchemicalMezeRecipe(sampling_time=0)


def test_alchemical_recipe_negative_sampling_time_raises():
    with pytest.raises(ValueError, match="greater than 0"):
        AlchemicalMezeRecipe(sampling_time=-4.0)


def test_zero_is_valid_for_other_bounded_fields():
    recipe = MezeRecipe(pressure=0.0, temperature=0.0, nb_cutoff=0.0)
    assert recipe.pressure.atm().value() == pytest.approx(0.0)
    assert recipe.temperature.kelvin().value() == pytest.approx(0.0)
    assert recipe.nb_cutoff.angstroms().value() == pytest.approx(0.0)


def test_alchemical_recipe_dt_coerced_to_picoseconds():
    recipe = AlchemicalMezeRecipe(dt=0.002)
    assert isinstance(recipe.dt, bss.Types.Time)


def test_alchemical_recipe_negative_dt_raises():
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        AlchemicalMezeRecipe(dt=-0.002)


# ---------------------------------------------------------------------------
# MezeRecipe.__getitem__ / __setitem__ / to_json
# ---------------------------------------------------------------------------

def test_recipe_getitem():
    recipe = MezeRecipe(group_name="vim2")
    assert recipe["group_name"] == "vim2"


def test_recipe_setitem():
    recipe = MezeRecipe()
    recipe["group_name"] = "kpc2"
    assert recipe.group_name == "kpc2"


def test_recipe_to_json_round_trip(tmp_path):
    recipe = MezeRecipe(group_name="vim2", metal="ZN")
    out_file = tmp_path / "recipe.json"

    recipe.to_json(str(out_file))

    with open(out_file) as f:
        data = json.load(f)
    assert data["group_name"] == "vim2"
    assert data["metal"] == "ZN"
