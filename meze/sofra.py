import warnings
import logging
warnings.filterwarnings("ignore", message="to-Python converter for std::__1::vector")
logging.getLogger("numexpr.utils").setLevel(logging.ERROR)
logging.getLogger("MDAnalysis").setLevel(logging.ERROR)
from dataclasses import dataclass
import dataclasses
import numpy as np
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
import MDAnalysis.analysis.distances
from MDAnalysis.core.groups import Residue as mdaResidue
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
    coordination_cut_off: float = Field(
        2.8, ge=0, description="Metal coordination cutoff in Å"
    )
    path_to_engine: Optional[str] = Field(
        None, description="Path to the MD engine executable (e.g. pmemd.cuda)"
    )
    def __str__(self) -> str:
        """Print recipe information as JSON
        """
        return self.model_dump_json(indent=4, fallback=str, warnings="none")

class ColdMezeRecipe(MezeRecipe):
    """Meze workflow recipe for minimisation and equilibration
    """
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

class HotMezeRecipe(MezeRecipe):
    """Meze workflow recipe for production runs
    """
    nb_cutoff: float = Field(
        12.0, ge=0, description="Cut-off for electrostatics interactions"
    )
    runtime: float = Field(
        100.0, description="Simulation time in nanoseconds"
    )
    dt: float = Field(
        0.002, description="Integrator timestep, in picoseconds"
    )
    temperature: float = Field(
        300.0,  description="Simulation temperature in kelvin"
    )
    pressure: float = Field(
        1.0, description="Simulation pressure in atm"
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

    def build_distance_restraints(
            self,
            force_constant: Optional[float] = 100.0,
            flat_bottom_radius: Optional[float] = 1.00
    ) -> dict[tuple[int, int], tuple[float, float, float]]:

        restraints = {}
        for metal_id, ligating_atoms in self.coordinating_residues.items():
            atom_group_1 = self.metals.select_atoms(f"bynum {metal_id}")

            for ligating_atom in ligating_atoms:
                key = (metal_id, ligating_atom.id)
                atom_group_2 = self.universe.select_atoms(
                    f"resid {ligating_atom.resid} and name {ligating_atom.name}"
                )
                distance = MDAnalysis.analysis.distances.dist(
                    atom_group_1, atom_group_2
                )[-1][0]
                restraints[key] = (
                    np.round(distance, 2),
                    np.round(force_constant, 2),
                    np.round(flat_bottom_radius, 2)
                )
        return restraints




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
            selection = f"element O or element N or element S" + \
            f" and sphzone {cutoff} (resid {self.metal_resids[i]})"
            ligands = self.universe.select_atoms(selection)
            key = self.metal_atomids[i] 
            metal_ligands[key] = ligands
        return metal_ligands

    def _setup_bss_system(self):
        self.system = bss.IO.readMolecules(
            [self.topology, self.coordinates]
        )

    def _run(
            self,
            system: Optional[bssSystem], 
            recipe: MezeRecipe,
            protocol: bss.Protocol,
            process_name: Optional[str] = "meze-run",
            config_options: Optional[dict] = None,
            namelist_options: Optional[list] = [],
            is_gpu: bool = True,
    ):
        input_system = system or self.system
        run_directory = os.path.join(recipe.workdir, process_name)
        os.makedirs(run_directory, exist_ok=True)

        process = bss.Process.Amber(
            system = input_system,
            protocol = protocol,
            work_dir=run_directory,
            name=process_name,
            extra_options=config_options,
            extra_lines=namelist_options,
            is_gpu=is_gpu,
            exe=recipe.path_to_engine
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
    

    def get_active_site_atom_group(self) -> mda.AtomGroup:
        """Get active site based on metal and coordination cutoff

        Returns:
            mda.AtomGroup: metal and residues in its coordination sphere
        """
        coordinating_residues = self.universe.select_atoms("")
        for metal_id in self.metal_atomids:
            coordinating_residues += self.coordinating_residues[metal_id]
        return self.metals + coordinating_residues
    

@dataclass
class ColdMeze(Meze):
    recipe: ColdMezeRecipe
    exclude_resids: Optional[Union[int, list[int]]] = None

    @classmethod
    def from_files(
        cls, 
        topology: str, 
        coordinates: str, 
        exclude_resids: Optional[Union[int, list[int]]] = None,
        **kwargs
    ) -> "ColdMeze":
        """
        Build a ColdMeze object from topology and coordinates.
        Passes extra kwargs into ColdMezeRecipe.
        """
        recipe = ColdMezeRecipe(**kwargs)
        return cls(
            topology=topology, 
            coordinates=coordinates, 
            exclude_resids=exclude_resids,
            recipe=recipe
        )
    
    def _build_restraint_mask(
            self, 
            position_restraints: str, 
            exclude_resids: Optional[Union[int, list[int]]] = None
    ) -> str | None:
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
        
        exclude_resids = exclude_resids or self.exclude_resids
        if isinstance(exclude_resids, int):
            exclude_resids = [exclude_resids]
        exclude_resids = set(exclude_resids or [])

        coordinating_atomgroups = next(iter(self.coordinating_residues.values()))
        for atomgroup in list(self.coordinating_residues.values())[1:]:
            coordinating_atomgroups += atomgroup

        coordinating_resids = [
            atom.resnum for atom in coordinating_atomgroups
            if atom.resnum not in exclude_resids
        ]
        
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

        config_options = {
            "cut": recipe.nb_cutoff,
            "ntpr": 1000,
            "iwrap": 0
        }
        
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
            if recipe.start_temperature != recipe.end_temperature:
                temperature = None
            else:
                temperature = bss.Types.Temperature(recipe.temperature, "K")

            protocol = bss.Protocol.Equilibration(
                timestep=bss.Types.Time(recipe.dt, "ps"),
                runtime=bss.Types.Time(recipe.runtime, "ps"),
                temperature_start=bss.Types.Temperature(recipe.start_temperature, "K"),
                temperature_end=bss.Types.Temperature(recipe.end_temperature, "K"),
                temperature=temperature,
                pressure=None,
                restraint="all" if position_restraints else None,
                force_constant=recipe.restraint_weight
            )
        elif protocol_type == "npt":
            config_options["barostat"] = recipe.barostat
            protocol = bss.Protocol.Equilibration(
                timestep=bss.Types.Time(recipe.dt, "ps"),
                runtime=bss.Types.Time(recipe.runtime, "ps"),
                temperature=bss.Types.Temperature(recipe.temperature, "K"),
                pressure=bss.Types.Pressure(recipe.pressure, "atm"),
                restraint="all" if position_restraints else None,
                force_constant=recipe.restraint_weight
            )
        else:
            raise ValueError(
                f"Invalid protocol type '{protocol_type}'. "
                f"Must be one of {allowed}."
            )
        
        return super()._run(
            protocol=protocol,
            recipe=recipe,
            system=system,
            process_name=process_name,
            config_options=config_options,
            is_gpu=is_gpu,
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

class HotMeze(Meze):
    recipe: HotMezeRecipe

    @classmethod
    def from_files(cls, topology: str, coordinates: str, **kwargs) -> "HotMeze":
        """
        Build a ColdMeze object from topology and coordinates.
        Passes extra kwargs into ColdMezeRecipe.
        """
        recipe = HotMezeRecipe(**kwargs)
        return cls(topology=topology, coordinates=coordinates, recipe=recipe)

    def run(
            self,
            workdir: Optional[str],
            system: Optional[bssSystem] = None,
            process_name: Optional[str] = "meze-run",
            nb_cutoff: Optional[float] = None,
            timestep: Optional[Union[float, bssTime]] = None,
            runtime: Optional[Union[float, bssTime]] = None,
            temperature: Optional[Union[float, bssTemperature]] = 300,
            pressure: Optional[Union[float, bssPressure]] = 1,
            engine_executable: Optional[str] = None,
            write_frequency: Optional[int] = 500000
    ):
        recipe = HotMezeRecipe(
            workdir=workdir or self.recipe.workdir,
            nb_cutoff=nb_cutoff or self.recipe.nb_cutoff,
            runtime= runtime or self.recipe.runtime,
            dt=timestep or self.recipe.dt,
            temperature=temperature or self.recipe.temperature,
            pressure=pressure or self.recipe.pressure,
            path_to_engine=engine_executable or self.recipe.path_to_engine
        )
        config_options = {"cut": recipe.nb_cutoff,
                          "ntpr": write_frequency,
                          "ntwx": write_frequency,
                          "ntwr": write_frequency,
                          "irest": 1,
                          "ntx": 5, 
                          "iwrap": 0}
        
        protocol = bss.Protocol.Production(
            timestep=bss.Types.Time(recipe.dt, "ps"),
            runtime=bss.Types.Time(recipe.runtime, "ns"),
            temperature=bss.Types.Temperature(recipe.temperature, "K"),
            pressure=bss.Types.Pressure(recipe.pressure, "atm")
        )

        return super()._run(
            protocol=protocol,
            recipe=recipe,
            system=system,
            process_name=process_name,
            config_options=config_options
        )
        

class QuantumMeze(Meze):
    recipe: MezeRecipe

    def __post_init__(self):
        super().__post_init__()
        self.qm_region = self._define_qm_region()
        self.qm_charge = self._get_qm_charge()

    @classmethod
    def from_files(cls, topology: str, coordinates: str, **kwargs) -> "QuantumMeze":
        """
        Build a Meze object from topology and coordinates.
        Passes extra kwargs into MezeRecipe.
        """
        recipe = MezeRecipe(**kwargs)
        return cls(topology=topology, coordinates=coordinates, recipe=recipe)

    def _define_qm_region(self) -> dict[str, list]:
        """Get a simple QM region 

        Returns:
            dict[str, list]: QM region split into a list of whole residues and atom ids
        """
        qm_region = {"whole_residues": [],
                     "atom_ids": []}
        protein = self.universe.select_atoms("protein")

        for metal_id, metal_ligands in self.coordinating_residues.items():
            qm_region["atom_ids"].append(metal_id)
            for residue in metal_ligands.residues:
                if residue in protein.residues:
                    side_chain_atoms = self._get_side_chain_selection(residue)
                    qm_region["atom_ids"].append(side_chain_atoms)
                else:
                    qm_region["whole_residues"].append(residue.resid)
        return qm_region
    

    def _get_side_chain_selection(self, residue: mdaResidue):
        """Return amino acid side chain atom indices

        Args:
            residue (mdaResidue): amino acid residue

        Returns:
            str: atom selection in the format {first_atom}-{last_atom}
        """
        n_terminus = "name N or name H"
        alpha_carbon = "name CA or name HA"
        c_terminus = "name C or name O"
        atoms_in_residue = self.universe.select_atoms(f"resid {residue.resid}")
        qm_region_for_residue = list(
            atoms_in_residue.select_atoms(
                f"not ({n_terminus} or {alpha_carbon} or {c_terminus})").ids
        )
        return f"{qm_region_for_residue[0]}-{qm_region_for_residue[-1]}"
    
    def _get_qm_charge(self):

        protein = self.universe.select_atoms("protein")
        qm_region = []
        active_site_residues = list(set([atom.residue for atom in self.get_active_site_atom_group()]))
        for residue in active_site_residues:
            if residue in protein.residues:
                n_terminus = "name N or name H"
                alpha_carbon = "name CA or name HA"
                c_terminus = "name C or name O"
                atoms_in_residue = self.universe.select_atoms(f"resid {residue.resid}")
                qm_region_for_residue = atoms_in_residue.select_atoms(f"not ({n_terminus} or {alpha_carbon} or {c_terminus})")
                qm_region.append(qm_region_for_residue)
            else:
                qm_residue = self.universe.select_atoms(f"resid {residue.resid}")
                qm_region.append(qm_residue)
        return sum([round(atom_group.charges.sum()) for atom_group in qm_region])
    
    def _write_qm_namelist(self, qm_theory: str = "DFTB3"):

        parsed_whole_residues = residue_restraint_mask(self.qm_region["whole_residues"])
        atom_ids = ",".join(list(map(str, self.qm_region["atom_ids"])))
        
        qm_config_options = {
            "qmmask": f"':{parsed_whole_residues}|(@{atom_ids})'",
            "writepdb": "1",
            "qmcharge": str(self.qm_charge),
            "qm_theory": f"'{qm_theory}'",
            "qmshake": "0",
            "qm_ewald": "1",
            "qm_pme": "1"
        }
        qm_namelist = [f"  {key}={value}" for key, value in qm_config_options.items()]
        qm_namelist.insert(0, "&qmmm")
        qm_namelist.append("/")
        return qm_namelist


class ColdQuantumMeze(QuantumMeze):
    recipe: ColdMezeRecipe

    @classmethod
    def from_files(cls, topology: str, coordinates: str, **kwargs) -> "ColdQuantumMeze":
        """
        Build a Meze object from topology and coordinates.
        Passes extra kwargs into MezeRecipe.
        """
        recipe = ColdMezeRecipe(**kwargs)
        return cls(topology=topology, coordinates=coordinates, recipe=recipe)

    def run(self,
            protocol_type: Literal["minimisation", "nvt", "npt"],
            system: Optional[bssSystem],
            workdir: Optional[str],
            restart: Optional[bool] = False,
            process_name: Optional[str] = "qm-meze-run",
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
            engine_executable: Optional[str] = None,
            qm_theory: Optional[str] = "DFTB3"
    ) -> "ColdQuantumMeze":
        
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
            path_to_engine=engine_executable or self.recipe.path_to_engine
        )

        qm_namelist = self._write_qm_namelist()

        config_options = {
            "cut": recipe.nb_cutoff,
            "ntpr": 50,
            "ntwx": 50,
            "ntwx": 50,
            "iwrap": 0,
            "ifqnt": 1
        }
        
        if restart:
            config_options["irest"] = 1
            config_options["ntx"] = 5

        allowed = ["minimisation", "nvt", "npt"]
        if protocol_type == "minimisation":
            config_options["ntmin"] = recipe.min_method
            config_options["maxcyc"] = recipe.max_cycles
            config_options["ncyc"] = recipe.n_sd_cycles
            protocol = bss.Protocol.Minimisation(
                steps=recipe.max_cycles, 
            )

        elif protocol_type == "nvt":
            if recipe.start_temperature != recipe.end_temperature:
                temperature = None
            else:
                temperature = bss.Types.Temperature(recipe.temperature, "K")

            protocol = bss.Protocol.Equilibration(
                timestep=bss.Types.Time(recipe.dt, "ps"),
                runtime=bss.Types.Time(recipe.runtime, "ps"),
                temperature_start=bss.Types.Temperature(recipe.start_temperature, "K"),
                temperature_end=bss.Types.Temperature(recipe.end_temperature, "K"),
                temperature=temperature,
                pressure=None,
            )
        elif protocol_type == "npt":
            raise NotImplementedError(
                f"Protocol type '{protocol_type}' not supported yet."
                f"Must be one of {allowed}"
            )
        else:
            raise ValueError(
                f"Invalid protocol type '{protocol_type}'. "
                f"Must be one of {allowed}."
            )
        return super()._run(
            protocol=protocol,
            recipe=recipe,
            system=system,
            process_name=process_name,
            config_options=config_options,
            namelist_options=qm_namelist,
            is_gpu=False,
        )
    
    def minimise(
            self,
            system: Optional[bssSystem] = None,
            workdir: Optional[str] = None,
            process_name: Optional[str] = "qm-min",
            max_cycles: Optional[int] = None,
            method: Optional[int] = None,
            n_sd_cycles: Optional[int] = None,
            nb_cutoff: Optional[float] = None,
            engine_executable: Optional[str] = None
    ) -> "ColdQuantumMeze":  
        
        return self.run(
            protocol_type="minimisation",
            system=system,
            workdir=workdir,
            process_name=process_name,
            max_cycles=max_cycles,
            n_sd_cycles=n_sd_cycles,
            nb_cutoff=nb_cutoff,
            method=method,
            engine_executable=engine_executable
        )
    
    def heat(
            self,
            system: Optional[bssSystem] = None,
            workdir: Optional[str] = None,
            restart: Optional[bool] = False,
            timestep: Optional[Union[float, bssTemperature]] = None,
            runtime: Optional[Union[float, bssTime]] = None,
            temperature: Optional[Union[float, bssTemperature]] = None,
            start_temperature: Optional[Union[float, bssTemperature]] = 300,
            end_temperature: Optional[Union[float, bssTemperature]] = 300,
            process_name: Optional[str] = "qm-nvt",
            engine_executable: Optional[str] = None
    ) -> "ColdQuantumMeze":

        return self.run(
            protocol_type="nvt",
            system=system,
            workdir=workdir,
            restart=restart,
            process_name=process_name,
            timestep=timestep,
            temperature=temperature,
            runtime=runtime,
            start_temperature=start_temperature,
            end_temperature=end_temperature,
            engine_executable=engine_executable
        )
    
class HotQuantumMeze(QuantumMeze):
    recipe: HotMezeRecipe

    @classmethod
    def from_files(cls, topology: str, coordinates: str, **kwargs) -> "HotQuantumMeze":
        """
        Build a Meze object from topology and coordinates.
        Passes extra kwargs into MezeRecipe.
        """
        recipe = HotMezeRecipe(**kwargs)
        return cls(topology=topology, coordinates=coordinates, recipe=recipe)
    
    def run(
            self,
            workdir: Optional[str],
            system: Optional[bssSystem] = None,
            process_name: Optional[str] = "qm-meze-run",
            nb_cutoff: Optional[float] = None,
            timestep: Optional[Union[float, bssTime]] = None,
            runtime: Optional[Union[float, bssTime]] = None,
            temperature: Optional[Union[float, bssTemperature]] = 300,
            pressure: Optional[Union[float, bssPressure]] = 1,
            engine_executable: Optional[str] = None,
            write_frequency: Optional[int] = 500,
            qm_theory: Optional[str] = "DFTB3",
            metal_resids_for_distance_restraints: Optional[Union[int, list[int]]] = None
    ) -> "HotQuantumMeze":
        
        recipe = HotMezeRecipe(
            workdir=workdir or self.recipe.workdir,
            nb_cutoff=nb_cutoff or self.recipe.nb_cutoff,
            runtime= runtime or self.recipe.runtime,
            dt=timestep or self.recipe.dt,
            temperature=temperature or self.recipe.temperature,
            pressure=pressure or self.recipe.pressure,
            path_to_engine=engine_executable or self.recipe.path_to_engine
        )

        config_options = {
            "cut": recipe.nb_cutoff,
            "ntpr": write_frequency,
            "ntwx": write_frequency,
            "ntwx": write_frequency,
            "iwrap": 0,
            "irest": 1,
            "ntx": 5,
            "ifqnt": 1
        }
        
        qm_namelist = self._write_qm_namelist(qm_theory)

        if metal_resids_for_distance_restraints:
            pass

        protocol = bss.Protocol.Production(
            timestep=bss.Types.Time(recipe.dt, "ps"),
            runtime=bss.Types.Time(recipe.runtime, "ps"),
            temperature=bss.Types.Temperature(recipe.temperature, "K"),
            pressure=None
        )

        return super()._run(
            protocol=protocol,
            recipe=recipe,
            system=system,
            process_name=process_name,
            config_options=config_options,
            namelist_options=qm_namelist,
            is_gpu=False,
        )
