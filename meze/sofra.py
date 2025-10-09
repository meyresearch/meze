import warnings
import logging
warnings.filterwarnings("ignore", message="to-Python converter for std::__1::vector")
logging.getLogger("numexpr.utils").setLevel(logging.ERROR)
logging.getLogger("MDAnalysis").setLevel(logging.ERROR)
from dataclasses import dataclass
import dataclasses
from pydantic import (
    Field,
    BaseModel
)
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

class MezeRecipe(BaseModel):
    """Meze workflow recipe
    """
    workdir: str = Field(default_factory=os.getcwd, description="Working directory")
    metal: str = Field("ZN", description="Metal resname")
    group_name: str = Field("meze", description="Group name for project")
    coordination_cut_off: float = Field(2.8, ge=0, description="Metal coordination cutoff in Å")
    
    def __str__(self) -> str:
        """Print recipe information as JSON
        """
        return self.model_dump_json(indent=4, fallback=str, warnings="none")

class ColdMezeRecipe(MezeRecipe):
    """Meze workflow recipe for minimisation and equilibration
    """
    path_to_engine: Optional[str] = Field(
        None, description="Path to the MD engine executable (e.g. pmemd.cuda)"
    )
    max_cycles: int = Field(1000, ge=0, description="Number of minimisation cycles")
    n_sd_cycles: int = Field(
        1000, ge=0, description="Number of steepest descent cycles (if min_method=1)"
    ) 
    min_method: int = Field(
        1, ge=0, description="Run steepest descent for n_sd_cycles, then conjugate gradient"
    )
    barostat: int = Field(
        2, ge=1, le=2, description="Type of barostat, 1: Berendsen, 2: MC"
    )
    nb_cutoff: float = Field(
        12.0, ge=0, description="Cut-off for electrostatics interactions"
    )
    runtime: float = Field(
        100.0, description="Simulation time in picoseconds"
    )
    dt: float = Field(
        0.002, description="Integrator timestep, in picoseconds"
    )
    start_temperature: float = Field(
        300.0, description="Simulation start temperature in kelvin"
    )
    end_temperature: float = Field(
        300.0, description="Simulation end temperature in kelvin"
    )
    temperature: float = Field(
        300.0,  description="Simulation temperature in kelvin"
    )
    pressure: float = Field(
        1.0, description="Simulation pressure in atm"
    )
    restraint_weight: float = Field(
        100.0, ge=0, description="Force constant for positional restraints in kcal/(mol*Å^2)"
    )

@dataclass
class Meze:
    topology: str 
    coordinates: str 
    recipe: MezeRecipe 

    def __post_init__(self):
        coordinate_extension = os.path.splitext(self.coordinates)[1]
        if coordinate_extension in [".rst7"]:
            coordinate_format = "RESTRT"
        else:
            coordinate_format = None
        self.universe = mda.Universe(
            self.topology,
            self.coordinates,
            topology_format="PARM7",
            format=coordinate_format
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
        recipe = MezeRecipe(**kwargs)
        return cls(topology=topology, coordinates=coordinates, recipe=recipe)

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
            selection = f"not (name {self.metal_resname} or element H)" + \
            f" and sphzone {cutoff} (resid {self.metal_resids[i]})"
            ligands = self.universe.select_atoms(selection)
            key = self.metal_atomids[i] 
            metal_ligands[key] = ligands
        return metal_ligands

    def _setup_bss_system(self):
        self.system = bss.IO.readMolecules(
            [self.topology, self.coordinates]
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
        recipe = ColdMezeRecipe(**kwargs)
        return cls(topology=topology, coordinates=coordinates, recipe=recipe)
    
    def _build_restraint_mask(self, position_restraints: str) -> str | None:
        """Build an amber-compatible restraint mask

        Args:
            position_restraints (str): what type of position restraints to apply

        Raises:
            ValueError: If position_restraint option is invalid.

        Returns:
            str: amber-style restraintmask
        """
        allowed = {None, "solute", "backbone", "metal-coordination"}
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
            return f"':{residue_restraint_mask(constraint_resids)}'"
        elif position_restraints == "backbone":
            constraint_resids = coordinating_resids + self.metal_resids.tolist()
            return f"'(@N,CA,C,O & !:WAT)|:{residue_restraint_mask(constraint_resids)}'"
        elif position_restraints == "metal-coordination":
            constraint_resids = coordinating_resids + self.metal_resids.tolist()
            return f"':{residue_restraint_mask(constraint_resids)}'"
        else: 
            return None
        
        

    def run(
            self,
            protocol_type: Literal["minimisation", "nvt", "npt"],
            system: Optional[bssSystem], 
            workdir: Optional[str],
            restart: Optional[bool] = False,
            position_restraints: Optional[str] = None,
            restraint_weight: Optional[float] = None,
            process_name: Optional[str] = "meze-run",
            max_cycles: Optional[int] = None,
            method: Optional[int] = None,
            barostat: Optional[int] = None,
            n_sd_cycles: Optional[int] = None,
            nb_cutoff: Optional[float] = None,
            timestep: Optional[Union[float, bssTime]] = None,
            runtime: Optional[Union[float, bssTime]] = None,
            temperature: Optional[Union[float, bssTemperature]] = None,
            start_temperature: Optional[Union[float, bssTemperature]] = 300,
            end_temperature: Optional[Union[float, bssTemperature]] = 300,
            pressure: Optional[Union[float, bssPressure]] = None,
            is_gpu: Optional[bool] = True,
            engine_executable: Optional["str"] = None
    ) -> "ColdMeze":

        recipe = ColdMezeRecipe(
            workdir=workdir or self.recipe.workdir,
            max_cycles=max_cycles or self.recipe.max_cycles,
            n_sd_cycles=n_sd_cycles or self.recipe.n_sd_cycles,
            min_method=method or self.recipe.min_method,
            barostat=barostat or self.recipe.barostat,
            nb_cutoff=nb_cutoff or self.recipe.nb_cutoff,
            runtime=runtime or self.recipe.runtime,
            dt=timestep or self.recipe.dt,
            temperature=temperature or self.recipe.temperature,
            start_temperature=start_temperature or self.recipe.start_temperature,
            end_temperature=end_temperature or self.recipe.end_temperature,
            pressure=pressure or self.recipe.pressure,
            restraint_weight=restraint_weight or self.recipe.restraint_weight,
            path_to_engine=engine_executable or self.recipe.path_to_engine
        )

        input_system = system or self.system
        run_directory = os.path.join(recipe.workdir, process_name)
        os.makedirs(run_directory, exist_ok=True)
        runtime = bss.Types.Time(recipe.runtime, "ps")
        dt = bss.Types.Time(recipe.dt, "ps")
        temperature = bss.Types.Temperature(recipe.temperature, "K")
        start_temperature = bss.Types.Temperature(recipe.start_temperature, "K")
        end_temperature = bss.Types.Temperature(recipe.end_temperature, "K")
        pressure = bss.Types.Pressure(recipe.pressure, "atm")

        config_options = {"cut": recipe.nb_cutoff,
                          "ntpr": 1000}
        
        if restart:
            config_options["irest"] = 1
            config_options["ntx"] = 5

        if position_restraints:
            config_options["restraintmask"] = self._build_restraint_mask(position_restraints)
        
        allowed = ["minimisation", "nvt", "npt"]
        if protocol_type == "minimisation":
            config_options["ntmin"] = recipe.min_method
            config_options["maxcyc"] = recipe.max_cycles
            config_options["ncyc"] = recipe.n_sd_cycles
            protocol = bss.Protocol.Minimisation(
                steps=recipe.max_cycles, 
                force_constant=recipe.restraint_weight,
                restraint="all" if position_restraints else None
            )
        elif protocol_type == "nvt":
            pressure = None
            if recipe.start_temperature != recipe.end_temperature:
                temperature = None
            protocol = bss.Protocol.Equilibration(
                timestep=dt,
                runtime=runtime,
                temperature_start=start_temperature,
                temperature_end=end_temperature,
                temperature=temperature,
                pressure=pressure,
                restraint="all" if position_restraints else None,
                force_constant=recipe.restraint_weight
            )
        elif protocol_type == "npt":
            config_options["barostat"] = recipe.barostat
            protocol = bss.Protocol.Equilibration(
                timestep=dt,
                runtime=runtime,
                temperature=temperature,
                pressure=pressure,
                restraint="all" if position_restraints else None,
                force_constant=recipe.restraint_weight
            )
        else:
            raise ValueError(
                f"Invalid protocol type '{protocol_type}'. "
                f"Must be one of {allowed}."
            )
    
        process = bss.Process.Amber(
            system = input_system,
            protocol = protocol,
            work_dir=run_directory,
            name=process_name,
            extra_options=config_options,
            is_gpu=is_gpu,
            exe=self.recipe.path_to_engine
        )

        process.start()
        process.wait()

        new_system = process.getSystem()
        topology, new_coordinates = bss.IO.saveMolecules(
            f"{run_directory}/next", system=new_system, fileformat=["prm7", "rst7"]
        )

        return dataclasses.replace(
            self,
            topology=topology,
            coordinates=new_coordinates,
            recipe=recipe
        )
    

    def minimise(
            self,
            system: Optional[bssSystem] = None,
            workdir: Optional[str] = None,
            position_restraints: Optional[
                Literal["solute", "backbone", "metal-coordination"]
            ] = None,
            restraint_weight: Optional[float] = None,
            process_name: Optional[str] = "min",
            max_cycles: Optional[int] = None,
            method: Optional[int] = None,
            n_sd_cycles: Optional[int] = None,
            nb_cutoff: Optional[float] = None,
            is_gpu: Optional[bool] = False,
            engine_executable: Optional[str] = None
    ) -> "ColdMeze":  
        
        return self.run(
            protocol_type="minimisation",
            system=system,
            workdir=workdir,
            position_restraints=position_restraints,
            process_name=process_name,
            restraint_weight=restraint_weight,
            max_cycles=max_cycles,
            n_sd_cycles=n_sd_cycles,
            nb_cutoff=nb_cutoff,
            method=method,
            is_gpu=is_gpu,
            engine_executable=engine_executable
        )

    def heat(
            self,
            system: Optional[bssSystem] = None,
            workdir: Optional[str] = None,
            position_restraints: Optional[
                Literal["solute", "backbone", "metal-coordination"]
            ] = None,
            restart: Optional[bool] = False,
            restraint_weight: Optional[float] = None,
            timestep: Optional[Union[float, bssTemperature]] = None,
            runtime: Optional[Union[float, bssTime]] = None,
            temperature: Optional[Union[float, bssTemperature]] = None,
            start_temperature: Optional[Union[float, bssTemperature]] = 300,
            end_temperature: Optional[Union[float, bssTemperature]] = 300,
            process_name: Optional[str] = "nvt",
            is_gpu: Optional[bool] = True,
            engine_executable: Optional[str] = None
    ) -> "ColdMeze":

        return self.run(
            protocol_type="nvt",
            system=system,
            workdir=workdir,
            restart=restart,
            position_restraints=position_restraints,
            process_name=process_name,
            restraint_weight=restraint_weight,
            timestep=timestep,
            temperature=temperature,
            runtime=runtime,
            start_temperature=start_temperature,
            end_temperature=end_temperature,
            is_gpu=is_gpu,
            engine_executable=engine_executable
        )

    def pressurise(
            self,
            system: Optional[bssSystem] = None,
            workdir: Optional[str] = None,
            position_restraints: Optional[
                Literal["solute", "backbone", "metal-coordination"]
            ] = None,
            restart: Optional[bool] = False,
            restraint_weight: Optional[float] = None,
            timestep: Optional[Union[float, bssTemperature]] = None,
            runtime: Optional[Union[float, bssTime]] = None,
            temperature: Optional[Union[float, bssTemperature]] = 300,
            pressure: Optional[Union[float, bssPressure]] = 1.0,
            process_name: Optional[str] = "npt",
            is_gpu: Optional[bool] = True,
            engine_executable: Optional[str] = None
    ) -> "ColdMeze":

        return self.run(
            protocol_type="npt",
            system=system,
            workdir=workdir,
            restart=restart,
            position_restraints=position_restraints,
            process_name=process_name,
            restraint_weight=restraint_weight,
            timestep=timestep,
            temperature=temperature,
            runtime=runtime,
            pressure=pressure,
            is_gpu=is_gpu,
            engine_executable=engine_executable
        )
