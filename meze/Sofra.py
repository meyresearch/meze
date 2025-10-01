from pydantic.dataclasses import dataclass
from pydantic import (
    Field,
    PrivateAttr
)
from meze import Protein
from typing import (
    Optional
)
import os
import MDAnalysis as mda

@dataclass
class MezeRecipe:
    topology: str = Field(..., description="Path to topology file")
    coordinates: str = Field(..., description="Path to coordinate file")
    workdir: str = Field(default_factory=os.getcwd, description="Working directory")
    metal: str = Field("ZN", description="Metal resname")
    group_name: str = Field("meze", description="Group name for project")
    coordination_cut_off: float = Field(2.8, ge=0, description="Metal coordination cutoff in Å")

@dataclass
class Meze:
    recipe: MezeRecipe = Field(default_factory=MezeRecipe, description="Meze workflow recipe")

    def __post_init__(self):
        self.universe = mda.Universe(
            self.recipe.topology,
            self.recipe.coordinates,
            topology_format="prmtop"
        )
        self._set_metal()

    def _set_metal(self):
        metal = self.recipe.metal.upper()
        self.metals = self.universe.select_atoms(f"resname {metal}")
        if len(self.metals) == 0:
            raise ValueError(f"No atoms found for metal: {self.recipe.metal}")

    @classmethod
    def from_files(cls, topology: str, coordinates: str, **kwargs):
        """Construct Meze from Amber topology and coordinates

        Args:
            topology (str): path to prm7/prmtop topology file
            coordinates (str): path to rst7/inpcrd coordinate file

        Returns:
            Meze: Meze class object
        """
        recipe = MezeRecipe(
            topology=topology,
            coordinates=coordinates,
            **kwargs
        )
        return cls(recipe=recipe)
    
    def get_active_site(self) -> mda.AtomGroup:
        """Get active site based on metal and coordination cutoff

        Returns:
            mda.AtomGroup: metal and residues in its coordination sphere
        """
        cutoff = self.recipe.coordination_cut_off
        selection = f"resname {self.recipe.metal} or around {cutoff} (resname {self.recipe.metal})"
        return self.universe.select_atoms(selection)

