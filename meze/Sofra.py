from pydantic.dataclasses import dataclass
from pydantic import (
    Field,
    field_validator
)
from meze import protein
from typing import (
    Optional,
    Literal,
    Union
)
import os
import MDAnalysis as mda
import BioSimSpace as bss
from BioSimSpace._SireWrappers import System as bssSystem
from BioSimSpace.Types._time import Time as bssTime
from BioSimSpace.Types._temperature import Temperature as bssTemperature
from BioSimSpace.Types._pressure import Pressure as bssPressure
from .utils import residue_restraint_mask
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
    n_sd_cycles: int = Field(1000, ge=0, description="Number of steepest descent cycles (if min_method=1)") 
    min_method: int = Field(1, ge=0, description="Run steepest descent for n_sd_cycles, then conjugate gradient")
    nb_cutoff: float = Field(12.0, ge=0, description="Cut-off for electrostatics interactions")
    runtime: float = Field(
        default_factory=lambda: bss.Types.Time(100.0, bss.Units.Time.picosecond),
        description="Simulation time in picoseconds"
    )
    dt: float = Field(
        default_factory=lambda: bss.Types.Time(0.002, bss.Units.Time.picosecond), 
        description="Integrator timestep, in picoseconds"
    )
    temperature: float = Field(
        default_factory=lambda: bss.Types.Temperature.kelvin(300.0), 
        description="Simulation temperature in kelvin"
    )
    pressure: float = Field(
        default_factory=lambda: bss.Types.Pressure.atm(1.0), 
        description="Simulation pressure in atm"
    )
    restraint_weight: float = Field(
        100.0, ge=0, description="Force constant for positional restraints in kcal/(mol*Å^2)"
    )

    @field_validator
    def ensure_time(cls, x):
        if isinstance(x, (int, float)):
            return bss.Types.Time(x, bss.Units.Time.picosecond)
    
    @field_validator
    def ensure_temperature(cls, x):
        if isinstance(x, (int, float)):
            return bss.Types.Temperature(x, bss.Units.Temperature.kelvin)
        
    @field_validator
    def ensure_pressure(cls, x):
        if isinstance(x, (int, float)):
            return bss.Types.Pressure(x, bss.Units.Pressure.atm)


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
    
    def _build_restraint_mask(self, position_restraints: str) -> str:
        """Build an amber-compatible restraint mask

        Args:
            position_restraints (str): what type of position restraints to apply

        Raises:
            ValueError: If position_restraint option is invalid.

        Returns:
            str: amber-style restraintmask
        """
        allowed = {None, "solute", "backbone", "heavy", "metal-coordination"}
        if position_restraints not in allowed:
            raise ValueError(
                f"Invalid restraint option '{position_restraints}'. "
                f"Must be one of {allowed}."
            )
        
        coordinating_atomgroups = next(iter(self.coordinating_residues.values()))
        for atomgroup in list(self.coordinating_residues.values())[1:]:
            coordinating_atomgroups += atomgroup
        coordinating_resids = [atom.resnum for atom in coordinating_atomgroups]
        if position_restraints == "solute":
            protein_resids = [atom.resnum for atom in self.universe.select_atoms("protein")]
            constraint_resids = protein_resids + coordinating_resids + self.metal_resids.tolist()
        elif position_restraints == "":
            pass

        return f":{residue_restraint_mask(constraint_resids)}"
            


    def run(
            self,
            protocol_type: str,
            system: Optional[bssSystem], 
            workdir: Optional[str],
            position_restraints: Optional[str] = None,
            restraint_weight: Optional[float] = None,
            process_name: Optional[str] = "meze-run",
            timestep: Optional[Union[float, bssTemperature]] = None,
            runtime: Optional[Union[float, bssTime]] = None,
            temperature: Optional[Union[float, bssTemperature]] = None,
            start_temperature: Optional[Union[float, bssTemperature]] = 300,
            end_temperature: Optional[Union[float, bssTemperature]] = 300,
            pressure: Optional[Union[float, bssPressure]] = None
    ) -> bssSystem:

        input_system = system or self.system
        target_workdir = workdir or self.recipe.workdir
        run_directory = os.path.join(target_workdir, process_name)
        restraint_force_constant = restraint_weight or self.recipe.restraint_weight
        os.makedirs(run_directory, exist_ok=True)
        md_runtime = runtime or self.recipe.runtime
        md_timestep = timestep or self.recipe.dt
        run_temperature = temperature or self.recipe.temperature
        npt_pressure = pressure or self.recipe.pressure
        config_options = {
            "ntmin": self.recipe.min_method,
            "maxcyc": self.recipe.max_cycles,
            "ncyc": self.recipe.n_sd_cycles,
            "cut": self.recipe.nb_cutoff
        }
        
        if position_restraints:
            config_options["restraintmask"] = self._build_restraint_mask(position_restraints)

        if protocol_type == "minimisation":
            protocol = bss.Protocol.Minimisation(
                steps=self.recipe.max_cycles, 
                force_constant=restraint_force_constant,
                restraint="all" if position_restraints else None
            )
        elif protocol_type == "equilibration":
            protocol = bss.Protocol.Equilibration(
                timestep=md_timestep
            )
        else:
            pass
    
        process = bss.Process.Amber(
            system = input_system,
            protocol = protocol,
            work_dir=run_directory,
            name=process_name,
            extra_options=config_options,
        )
        process.start()
        process.wait()
        new_system = process.getSystem()
        self.system = new_system
        return new_system

    def minimise(
            self,
            system: Optional[bssSystem] = None,
            workdir: Optional[str] = None,
            position_restraints: Optional[
                Literal["solute", "backbone", "heavy", "metal-coordination"]
            ] = None,
            restraint_weight: Optional[float] = None
    ) -> bssSystem:  
        """Run a minimisation with Amber

        Args:
            system (Optional[bssSystem], optional): System to minimise. Defaults to None.
            workdir (Optional[str], optional): Working directory for minimisation. Defaults to None.
            position_restraints (Optional[ Literal['solute', 'backbone', 'heavy', 'metal-coordination', optional): 
                                Whether to use positional restraints. Defaults to None.
            restraint_weight (Optional[float], optional): Force constant for position restraints. Defaults to None.

        Returns:
            bssSystem: Minimised system.
        """
        return self.run(
            protocol_type="minimisation",
            system=system,
            workdir=workdir,
            position_restraints=position_restraints,
            process_name="min",
            restraint_weight=restraint_weight
        )

    def heat(
            self,
            system: Optional[bssSystem] = None,
            workdir: Optional[str] = None,
            position_restraints: Optional[
                Literal["solute", "backbone", "heavy", "metal-coordination"]
            ] = None,
            restraint_weight: Optional[float] = None,
            timestep: Optional[Union[float, bssTemperature]] = None,
            runtime: Optional[Union[float, bssTime]] = None,
            temperature: Optional[Union[float, bssTemperature]] = None,
            start_temperature: Optional[Union[float, bssTemperature]] = 300,
            end_temperature: Optional[Union[float, bssTemperature]] = 300,
            pressure: Optional[Union[float, bssPressure]] = None
    ) -> bssSystem:
        return self.run(
            protocol_type="equilibration",
            system=system,
            workdir=workdir,
            position_restraints=position_restraints,
            process_name="min",
            restraint_weight=restraint_weight,
        )
