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
import BioSimSpace as bss
@dataclass
class MezeRecipe:
    """Meze workflow recipe
    """
    topology: str = Field(..., description="Path to topology file")
    coordinates: str = Field(..., description="Path to coordinate file")
    workdir: str = Field(default_factory=os.getcwd, description="Working directory")
    metal: str = Field("ZN", description="Metal resname")
    group_name: str = Field("meze", description="Group name for project")
    coordination_cut_off: float = Field(2.8, ge=0, description="Metal coordination cutoff in Å")

@dataclass
class ColdMezeRecipe(MezeRecipe):
    """Meze workflow recipe for minimisation and equilibration
    """
    max_cycles: int = Field(1000, ge=0, description="Number of minimisation cycles")
    n_sd_cycles: int = Field(1000, ge=0, description="Number of steepest descent cycles (if ntmin=1)") 
    min_method: int = Field(1, ge=0, description="Run steepest descent for n_sd_cycles, then conjugate gradient")
    time: float = Field(100.0, gt=0, description="Simulation time in picoseconds")
    dt: float = Field(0.002, gt=0, description="Integrator timestep, in picoseconds")
    temperature: float = Field(300.0, ge=0, description="Simulation temperature in kelvin")
    pressure: float = Field(1.0, ge=0, description="Simulation pressure in atm")
    position_restraints: Optional[str] = None
    restraint_weight: float = Field(100.0, ge=0, description="Force constant for positional restraints in kcal/(mol*Å^2)")

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
        self.coordinating_residues = self._get_metal_coordinating_residues()
        self._setup_bss_system()

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

    def _set_metal(self):
        """Set metal residue names and indices based on MDAnalysis Universe

        Raises:
            ValueError: If no atoms matching to given metal name are found
        """
        metal = self.recipe.metal.upper()
        self.metals = self.universe.select_atoms(f"resname {metal}")
        if len(self.metals) == 0:
            raise ValueError(f"No atoms found for metal: {self.recipe.metal}")
        self.metal_resids = self.metals.resids
        self.metal_atomids = self.metals.atoms.ids
        self.metal_resname = metal

    def _get_metal_coordinating_residues(self) -> dict[int, mda.AtomGroup]:
        """Get residues coordinating to metal

        Returns:
            dict[int, mda.AtomGroup]: key: metal atom id, value: atom group of coordinating residues
        """
        cutoff = self.recipe.coordination_cut_off
        metal_ligands = {}
        for i in range(len(self.metal_resids)):
            ligands = self.universe.select_atoms(
                f"not (name {self.metal_resname} or element H) and sphzone {cutoff} (resid {self.metal_resids[i]})"
            )
            key = self.metal_atomids[i] 
            metal_ligands[key] = ligands
        return metal_ligands

    def _setup_bss_system(self):
        self.system = bss.IO.readMolecules(
            [self.recipe.topology, self.recipe.coordinates]
        )

    def get_active_site(self) -> mda.AtomGroup:
        """Get active site based on metal and coordination cutoff

        Returns:
            mda.AtomGroup: metal and residues in its coordination sphere
        """
        cutoff = self.recipe.coordination_cut_off
        selection = f"resname {self.recipe.metal} or around {cutoff} (resname {self.recipe.metal})"
        return self.universe.select_atoms(selection)

    def run(
            self,
            system
    ):
        pass
@dataclass
class ColdMeze(Meze):
    recipe: ColdMezeRecipe

    @classmethod
    def from_files(cls, topology: str, coordinates: str, **kwargs) -> "ColdMeze":
        """
        Build a ColdMeze object from topology and coordinates.
        Passes extra kwargs into ColdMezeRecipe.
        """
        recipe = ColdMezeRecipe(topology=topology, coordinates=coordinates, **kwargs)
        return cls(recipe=recipe)
    
    def minimise(self):
        protocol = bss.Protocol.Minimisation(
            steps=self.recipe.max_cycles,
            restraint=self.recipe.position_restraints,
            force_constant=self.recipe.restraint_weight
        )
        # self.run()
        print(protocol)

