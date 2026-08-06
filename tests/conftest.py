import pytest
import json
from meze import ColdMeze, ColdMezeRecipe

@pytest.fixture(scope="session")
def vim2_recipe_json():
    with open("data/inputs/model_0/protein/vim2/model_0_recipe.json") as f:
        return json.load(f)

@pytest.fixture(scope="session")
def vim2_cold_meze(vim2_recipe_json):
    return ColdMeze.from_files(
        recipe=ColdMezeRecipe(**vim2_recipe_json),
        pdb_file="data/inputs/model_0/protein/vim2/vim2.fixed.pdb",
    )
