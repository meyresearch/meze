import glob 
import json
import warnings
import logging
warnings.filterwarnings("ignore", message="to-Python converter for std::__1::vector")
logging.getLogger("numexpr.utils").setLevel(logging.ERROR)
logging.getLogger("MDAnalysis").setLevel(logging.ERROR)
from dataclasses import (
    dataclass, 
    field
)
import dataclasses
import numpy as np
from pydantic import (
    Field,
    field_validator,
    BaseModel
)
from typing import (
    Any,
    List,
    Optional,
    Literal,
    Union,
    Self,
    List
)
import pickle
import pathlib
from .ligand import Ligand
import os
from pymsmt.mcpb.gene_final_frcmod_file import fcfit_ep_bond
import MDAnalysis as mda
import MDAnalysis.analysis.distances
from MDAnalysis.topology.guessers import guess_types
from MDAnalysis.core.groups import Residue as mdaResidue
import BioSimSpace as bss
from BioSimSpace._SireWrappers import System as bssSystem
from BioSimSpace.Types._time import Time as bssTime
from BioSimSpace.Types._temperature import Temperature as bssTemperature
from BioSimSpace.Types._pressure import Pressure as bssPressure
from .utils import (
    _residue_restraint_mask,
    _write_distance_restraints,
    _write_tleap_solvation_input,
    _write_gaussian_script,
    _pretty,
    _parse_mcpbpy_input,
    _check_log_files,
    _get_mol2_charge,
    _edit_mcpbpy_tleap_input
)
import shutil
from rich.logging import RichHandler
from rich.console import Console

console = Console(force_terminal=True, color_system="truecolor")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)],
    force=True,
)

log = logging.getLogger("rich")

class MezeRecipe(BaseModel):
    """Meze workflow recipe
    """
    workdir: str = Field(default_factory=os.getcwd, description="Working directory")
    metal: str = Field("ZN", description="Metal element")
    metal_charge: int = Field(2, description="Metal charge")
    group_name: str = Field("meze", description="Group name for project")
    coordination_cut_off: float = Field(
        2.8, ge=0, description="Metal coordination cutoff in Å"
    )
    path_to_engine: Optional[str] = Field(
        None, description="Path to the MD engine executable (e.g. pmemd.cuda)"
    )
    model: Optional[int] = Field(
        None, description="Metal modelling option"
    )
    gaussian_version: str = Field(
        "g16", description="Gaussian version"
    )
    memory: float = Field(12000, description="Memory for Gaussian calculations in MB")

    nprocshared: int = Field(8, description="Number of processors for Gaussian calculations")

    only_optimise_hydrogens: bool = Field(
        True, description="Only optimise hydrogen atoms"
    )
    protein_forcefield: str = Field(
        "ff14SB", description="Protein forcefield"
    )
    ligand_forcefield: str = Field(
        "gaff2", description="Ligand forcefield"
    )
    water_model: str = Field(
        "tip3p", description="Water model"
    )
    box_shape: str = Field(
        "octahedral", description="Box shape"
    )
    box_edges: float = Field(
        10.0, ge=0, description="Box edges in Å"
    )
    solvent_closeness: float = Field(
        0.75, ge=0, le=1, description="Solvent closeness"
    )

    @field_validator("model", mode="before")
    @classmethod
    def validate_model(cls, v):
        if v is None:
            return v
        try:
            return int(v)
        except (TypeError, ValueError):
            raise ValueError(f"Cannot covert model='{v}' to int")

    def __str__(self) -> str:
        """Print recipe information as JSON
        """
        return self.model_dump_json(indent=4, fallback=str, warnings="none")

    def to_json(self, file: str):
        with open(file, "w") as ofile:
            ofile.write(self.model_dump_json(indent=2))

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
        0.001, description="Integrator timestep, in picoseconds"
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
    disulfide_bridges: Optional[List[dict[str, int]]] = None
    ligand: Optional[Ligand] = None 
    ligand_resid: Optional[int] = None
    non_standard_residues: dict[dict] | List[Ligand] = field(default_factory=dict)   
    parameterisation_directory: Optional[str] = None
    mcpbpy_input_file: Optional[str] = None
    tleap_input_file: Optional[str] = None
    restraint_file: Optional[str] = None
    exclude_resids: Optional[Union[int, list[int]]] = field(default_factory=list)
    ligand_resname: Optional[str] = None

    def __post_init__(self):
        coordinate_extension = os.path.splitext(self.coordinates)[1]
        if coordinate_extension in [".rst7"]:
            coordinate_format = "RESTRT"
        else:
            coordinate_format = None
        topology_extension = os.path.splitext(self.topology)[1]
        try:
            if coordinate_extension == topology_extension:
                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.filterwarnings(
                        "always",
                        message=r"Unknown element.*empty element record",
                        category=UserWarning,
                        module=r"MDAnalysis\.topology\.PDBParser",
                    )
                    self.universe = mda.Universe(
                        self.topology,
                    )   
                    guessed_elements = guess_types(self.universe.atoms.names)
                    self.universe.add_TopologyAttr("elements", guessed_elements)
            else:         
                self.universe = mda.Universe(
                    self.topology,
                    self.coordinates,
                    topology_format="PARM7",
                    format=coordinate_format
                )

        except FileNotFoundError:
            print("Could not create meze object:\n")
            raise            

        self._set_metal()
        self.coordinating_residues = self._get_metal_coordinating_residues()
        self._setup_bss_system()
        
        if self.non_standard_residues and isinstance(self.non_standard_residues, dict):
            self._validate_non_standard_residues()
        
        if self.ligand and self.ligand.parameterised and not self.ligand_resid:
            self.ligand_resid = self.get_ligand_resid()
        elif not self.ligand and self.ligand_resname:
            log.warning(
                "Ligand not set by user, inferring from ligand residue name"
            )
            self._set_ligand()
        else:
            log.warning(
                "Ligand not set by user. Are you sure you want to continue without a ligand?"
            )


    def __str__(self) -> str:
        return _pretty(self)

    def save(self, filename: str):
        suffix = pathlib.Path(filename).suffix
        if not suffix:
            filename += ".pkl"
        with open(filename, "wb") as file:
            pickle.dump(self, file)
    
    def add_to_sofra(self, 
                     filename: str, 
                     key: str, 
                     extra_fields: Optional[dict] = None):
        new_entry = {
            key: {}
        }

        if self.parameterisation_directory is not None:
            new_entry[key]["parameterisation_directory"] = self.parameterisation_directory
        if self.mcpbpy_input_file is not None:
            new_entry[key]["mcpbpy_input_file"] = self.mcpbpy_input_file
        if self.tleap_input_file is not None:
            new_entry[key]["tleap_input_file"] = self.tleap_input_file
        if self.restraint_file is not None:
            new_entry[key]["restraint_file"] = self.restraint_file

        if extra_fields:
            new_entry[key].update(extra_fields)

        if os.path.exists(filename):
            with open(filename, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    log.warning(f"Could not decode JSON from {filename}:"
                                f"{e}")
                    data = {}
        else:
            data = {}

        data.update(new_entry)

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load(cls, filename: str):
        if not os.path.isfile(filename):
            raise FileNotFoundError(
                f"Pickle meze file not found: {filename}"
            )
        with open(filename, "rb") as file:
            return pickle.load(file)


    @classmethod
    def from_files(
        cls, 
        topology: str, 
        coordinates: str, 
        **kwargs
    ):
        """Construct Meze from Amber topology and coordinates

        Args:
            topology (str): path to prm7/prmtop topology file
            coordinates (str): path to rst7/inpcrd coordinate file

        Returns:
            Meze: Meze class object
        """
        recipe = MezeRecipe(**kwargs)
        return cls(
            topology=topology, 
            coordinates=coordinates,
            recipe=recipe
        )

    def _set_ligand(self):

        ag = self.universe.select_atoms(f"resname {self.ligand_resname}")
        if len(ag) == 0:
            log.warning(
                f"Could not find ligand with resname {self.ligand_resname}"
                "Ligand not set for system. Consider adding a ligand with meze.add_ligand()"
                "or use a pickle file to load in a meze object"
            )
        
        self.ligand = Ligand(
            file=[self.coordinates, self.topology],
            name=self.ligand_resname,
            charge=ag.charges.sum(),
            parameterised=True,
            residue_name=self.ligand_resname
        )
        
        

    def get_small_molecule_resname(self) -> str | None:

        selection = self.universe.select_atoms(
            "not protein and not water"
        )
        non_standard_residues = [
            "MOH", "DOH", "Na+", "CL-", "ASZ", "GLZ", "HDZ", "HEZ", "CYZ", "CYM",
        ]

        resname = None
        for residue in selection.residues:
            if (
                residue.resname != self.metal_resname.upper()
                and residue.resname not in non_standard_residues
            ):
                resname = residue.resname

        return resname


    def build_distance_restraints( 
            self,
            metal_atom_ids: Optional[list[int]] = None,
            coordinating_residues: Optional[dict[int, mda.AtomGroup]] = None,
            force_constant: Optional[float] = 100.0,
            flat_bottom_radius: Optional[float] = 1.00,
            exclude_residues: Optional[Union[int, list[int]]] = None
    ) -> dict[tuple[int, int], tuple[float, float, float]]:

        metal_atom_ids = metal_atom_ids or list(self.coordinating_residues.keys())
        ligand_residues = coordinating_residues or self.coordinating_residues
        exclude = self.exclude_resids or exclude_residues
        if isinstance(exclude, int):
            exclude = [exclude]
        
        if not exclude:
            exclude = []

        restraints = {}
        for metal_id, ligating_atoms in ligand_residues.items():
            if metal_id in metal_atom_ids:
                atom_group_1 = self.metals.select_atoms(f"id {metal_id}")

                for ligating_atom in ligating_atoms:
                    if ligating_atom.resid in exclude:
                        continue

                    if self.ligand and ligating_atom.resname.upper() == self.ligand.residue_name:
                        continue
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
    
    def _prepare_distance_restraints(
        self
    ) -> Optional[list[str]]:

        metal_atom_ids = [
            self.universe.select_atoms(f"resid {resid}").ids[0]
            for resid in self.metal_resids
        ]
        distance_restraints_dict = self.build_distance_restraints(metal_atom_ids)
        return _write_distance_restraints(distance_restraints_dict)

    def _prepare_angle_restraints(
        self
    ) -> Optional[list[str]]:

        metal_atom_ids = [
            self.universe.select_atoms(f"resid {resid}").ids[0]
            for resid in self.metal_resids
        ]
        angle_restraints_dict = self.build_angle_restraints(metal_atom_ids)
        return _write_distance_restraints(angle_restraints_dict)
    
    def build_angle_restraints(
            self,
            metal_atom_ids: Optional[list[int]] = None,
            force_constant: Optional[float] = 100.0,
            flat_bottom_radius: Optional[float] = 1.00,
            exclude_residues: Optional[Union[int, list[int]]] = None
    ) -> dict[tuple[int, int], tuple[float, float, float]]:
        """Enforce "angle" restraints through additional distance restraints between vertex atoms.
        """
        metal_atom_ids = metal_atom_ids or list(self.coordinating_residues.keys())
        exclude = self.exclude_resids or exclude_residues
        if isinstance(exclude, int):
            exclude = [exclude]
        
        if not exclude:
            exclude = []

        restraints = {}
        for metal_id, ligating_atoms in self.coordinating_residues.items():
            if metal_id in metal_atom_ids:
                vertices = []
                for ligating_atom in ligating_atoms:
                    if ligating_atom.resid in exclude:
                        continue
                    if self.ligand and ligating_atom.resname.upper() == self.ligand.residue_name:
                        continue
                    if ligating_atom.id == metal_id:
                        continue
                    vertices.append(ligating_atom)

                n_vertices = len(vertices)
                for i in range(n_vertices):
                    for j in range(i + 1, n_vertices):

                        atom_i = vertices[i]
                        atom_j = vertices[j]
                        if atom_i.resid == atom_j.resid:
                            continue

                        key = (atom_i.id, atom_j.id)
                        atom_group_1 = self.universe.select_atoms(
                            f"resid {atom_i.resid} and name {atom_i.name}" 
                        )
                        atom_group_2 = self.universe.select_atoms(
                            f"resid {atom_j.resid} and name {atom_j.name}"
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
        input_metal = self.recipe.metal
        if len(input_metal) == 1:
            metal = input_metal.upper()
            self.recipe.metal = metal
        elif len(input_metal) == 2:
            metal = f"{input_metal[0].upper()}{input_metal[1].lower()}"
            self.recipe.metal = metal 

        try:
            self.metals = self.universe.select_atoms(f"element {metal}")
        except AttributeError as e:
            if "elements" in str(e):
                log.warning(
                    "\nNo element information found in PDB file.\n"
                    "Guessing element information from atom names.\n"
                    "This may lead to incorrect identification of metal atoms.\n"
                    "Consider fixing your PDB file with e.g. pdb4amber.\n"
                )
                guessed_elements = guess_types(self.universe.atoms.names)
                self.universe.add_TopologyAttr("elements", guessed_elements)
                self.metals = self.universe.select_atoms(f"element {metal.upper()}")
            else:
                raise e

        if len(self.metals) == 0:
            self.metals = self.universe.select_atoms(f"element {metal.upper()}")
            if len(self.metals) == 0:
                raise ValueError(f"No atoms found for metal: {self.recipe.metal}")

        self.metal_resids = self.metals.resids
        self.metal_atomids = self.metals.atoms.ids
        self.metal_resname = metal
        self.metal_element = metal.capitalize()

    def _get_metal_coordinating_residues(self) -> dict[int, mda.AtomGroup]:
        """Get residues coordinating to metal

        Returns:
            dict[int, mda.AtomGroup]: key: metal atom id, value: atom group of coordinating residues
        """
        cutoff = self.recipe.coordination_cut_off
        metal_ligands = {}
        try:
            for i in range(len(self.metal_resids)):
                selection = f"element O or element N or element S" + \
                f" and sphzone {cutoff} (resid {self.metal_resids[i]})"
                ligands = self.universe.select_atoms(selection)
                key = self.metal_atomids[i] 
                metal_ligands[key] = ligands
        except AttributeError as e:
            if "elements" in str(e):
                log.warning(
                    "\nNo element information found in PDB file.\n"
                    "Guessing element information from atom names.\n"
                    "This may lead to incorrect identification of metal coordination.\n"
                    "Consider fixing your PDB file with e.g. pdb4amber.\n"
                )    
                guessed_elements = guess_types(self.universe.atoms.names)
                self.universe.add_TopologyAttr("elements", guessed_elements)
                selection = f"element O or element N or element S" + \
                f" and sphzone {cutoff} (resid {self.metal_resids[i]})"
                ligands = self.universe.select_atoms(selection)
                key = self.metal_atomids[i] 
                metal_ligands[key] = ligands
            else:
                raise e
        return metal_ligands

    def _setup_bss_system(self):
        self.system = bss.IO.readMolecules(
            [self.topology, self.coordinates]
        )

    def _validate_disulfide_bridges(self):   
        if not self.disulfide_bridges:
            return 
        with open(self.coordinates, "r") as pdb:
            pdb_lines = pdb.readlines()
        
        conect_lines = {}
        counter = 1
        for line in pdb_lines:
            if "CONECT" in line:
                parts = line.split()
                conect_lines[f"CONECT{counter}"] = [int(parts[1]), int(parts[2])]
                counter += 1  

        seen_bridges = set()

        for bridge in self.disulfide_bridges:
            if not {"resid1", "resid2"} <= bridge.keys():
                raise ValueError(f"Invalid disulfide bridge entry: {bridge}")
            
            r1, r2 = bridge["resid1"], bridge["resid2"]

            if r1 == r2:
                raise ValueError(f"Disulfide bridge cannot connect residue {r1} to itself.")

            pair = tuple(sorted((r1, r2)))

            if pair in seen_bridges:
                raise ValueError(f"Duplicate disulfide bridge: {pair}")
            seen_bridges.add(pair)

            try:
                cyx1 = self.universe.select_atoms(f"resid {r1}").residues[0]
                cyx2 = self.universe.select_atoms(f"resid {r2}").residues[0]

            except IndexError:
                raise ValueError(f"Residue {r1} or {r2} not found in structure.")

            if cyx1.resname != "CYX" or cyx2.resname != "CYX":
                raise ValueError(
                    f"Disulfide bonds require CYX residues. "
                    f"Got {cyx1.resname} and {cyx2.resname} for {r1} and {r2}."
                )
      
            sg1 = cyx1.atoms.select_atoms("name SG")
            sg2 = cyx2.atoms.select_atoms("name SG")

            if len(sg1) == 0 or len(sg2) == 0:
                raise ValueError(f"Missing SG atom in residues {r1} or {r2}.")
            
            _, _, dists = MDAnalysis.analysis.distances.dist(sg1, sg2)
            distance = dists[0]
            if distance > 3.0:
                raise ValueError(
                    f"Disulfide {r1}-{r2} too long: {distance:.2f} Å (likely incorrect)."
                )
            
            for _, ids in conect_lines.items():
                if sg1.ids[0] in ids or sg2.ids[0] in ids:
                    log.warning(
                        f"Residues {r1} and {r2} appear to already have a disulfide bond "
                        f"in the CONECT records."
                        f"No explicit bond will be added in tleap."
                    )
                    self.disulfide_bridges = None

    def _run(
            self,
            system: Optional[bssSystem], 
            recipe: MezeRecipe,
            protocol: bss.Protocol,
            process_name: Optional[str] = "meze-run",
            config_options: Optional[dict] = None,
            namelist_options: Optional[list] = None,
            is_gpu: bool = True,
            distance_restraints: Optional[list] = None,
            distance_write_frequency: Optional[int] = 100,
    ):
        input_system = system or self.system
        run_directory = os.path.join(recipe.workdir, process_name)
        os.makedirs(run_directory, exist_ok=True)
        
        namelist_options = namelist_options or []

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

        if self.recipe.model == 0:
            coordination_restraints = self._prepare_distance_restraints()
            angle_restraints = self._prepare_angle_restraints()
            distance_restraints = coordination_restraints + angle_restraints

        if distance_restraints:
            config_file = process._config_file
            restraint_file = os.path.join(recipe.workdir, "restraints.RST")

            if not os.path.isfile(restraint_file):
                with open(restraint_file, "w") as file:
                    file.writelines(distance_restraints)
            
            step_restraint_file = os.path.join(run_directory, "restraints.RST")
            shutil.copyfile(restraint_file, step_restraint_file)
            
            with open(config_file, "a") as file:
                file.write(f"&wt TYPE='DUMPFREQ', istep1={distance_write_frequency} /\n")

            if not namelist_options:
                with open(config_file, "a") as file:
                    file.write(f"&wt TYPE=\"END\", /\n")
        
            with open(config_file, "a") as file:
                file.write("\n")
                file.write(f"DISANG=restraints.RST\n")
                file.write(f"DUMPAVE=distances.out\n")


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
    
    def get_ligand_resid(self):
        return self.universe.select_atoms(f"resname {self.ligand.residue_name}").resids[0]

    def get_active_site_atom_group(self) -> mda.AtomGroup:
        """Get active site based on metal and coordination cutoff

        Returns:
            mda.AtomGroup: metal and residues in its coordination sphere
        """
        coordinating_residues = self.universe.select_atoms("")
        for metal_id in self.metal_atomids:
            coordinating_residues += self.coordinating_residues[metal_id]
        return self.metals + coordinating_residues
    
    def add_ligand(
            self, 
            ligand_file: Union[str, list[str]], 
            name: str | None = None,
            ligand_charge: Optional[int] = 0
    ) -> Self:
        ligand = Ligand(ligand_file, name=name, charge=ligand_charge)
        
        return dataclasses.replace(
            self,
            ligand=ligand
        )

    def _validate_non_standard_residues(self):
        for residue, properties in self.non_standard_residues.items():
            if not {"charge", "atom_type"} <= properties.keys():
                raise ValueError(
                    f"Non-standard residue '{residue}' must have 'charge' and 'atom_type' properties."
                )
            if not isinstance(properties["charge"], int):
                raise ValueError(
                    f"Non-standard residue '{residue}' has invalid 'charge': {properties['charge']}"
                )
            if properties["atom_type"] not in ["amber", "gaff", "gaff2"]:
                log.warning(
                     f"Non-standard residue '{residue}' has potentially unsupported 'atom_type': {properties['atom_type']}"
                )
                   
                

    def parameterise_non_standard_residues(
            self, 
            directory: str, 
            non_standard_parameterisation_method: Literal["antechamber", "tleap"] = "antechamber"
    ) -> Optional[list[Ligand]]:
        if self.non_standard_residues:
            
            for residue in self.non_standard_residues.keys():
                if not residue.isnumeric():
                    ag = self.universe.select_atoms(f"resname {residue}")
                    if len(ag) == 0:
                        raise RuntimeError(
                            f"Could not find residue with resname {residue}"
                        )
                else:
                    try:
                        residue = int(residue)
                        ag = self.universe.select_atoms(f"resid {residue}")
                    except ValueError as e:
                        log.error(
                            f"Could not convert residue id {residue} to integer:"
                            f"{e}"
                        )
                residue = str(residue)
                ag.write(f"{directory}/{residue}.pdb")
            
            non_standard_residues = [
                Ligand(
                    file=f"{directory}/{residue}.pdb",
                    name=residue,
                    charge=properties["charge"],
                    atom_type=properties["atom_type"]
                )
                for residue, properties in self.non_standard_residues.items()
            ]
            parameterised_non_standard_residues = [
                non_standard_residue.parameterise(
                    directory=directory,
                    atom_type=non_standard_residue.atom_type,
                    residue_name=non_standard_residue.name,
                    method=non_standard_parameterisation_method,
                    force_field=self.recipe.ligand_forcefield
                )
                for non_standard_residue in non_standard_residues
            ]
        else:
            parameterised_non_standard_residues = None
        
        return parameterised_non_standard_residues


    def add_water(
            self,
            directory: str | None = None, 
            mcpbpy_tleap_file: str | None = None, 
            non_standard_parameterisation_method: Literal["antechamber", "tleap"] = "antechamber"
    ) -> Self:
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        self._validate_disulfide_bridges()

        if self.recipe.model == 0: 
            if self.ligand:
                parameterised_ligand = self.ligand.parameterise(directory)
                ligand_name = parameterised_ligand.name 
            else: 
                parameterised_ligand = None
            if self.non_standard_residues:
                parameterised_non_standard_residues = self.parameterise_non_standard_residues(
                    directory=directory,
                    non_standard_parameterisation_method=non_standard_parameterisation_method
                )
        elif self.recipe.model == 2:
            parameterised_ligand = self.ligand 
            parameterised_non_standard_residues = self.non_standard_residues
        else: 
            raise NotImplementedError(
                f"Model option {self.recipe.model} is not implemented"
            )

        if not mcpbpy_tleap_file:
            tleap_input_file = os.path.join(directory, f"tleap_solvate.in")
            tleap_output_file = os.path.join(directory, f"tleap_solvate.out")

            tleap_lines = _write_tleap_solvation_input(
                protein_file=self.topology,
                ligand=parameterised_ligand,
                non_standard_residues=parameterised_non_standard_residues,
                disulfide_bridges=self.disulfide_bridges,
                protein_ff=self.recipe.protein_forcefield,
                ligand_ff=self.recipe.ligand_forcefield,
                water_model=self.recipe.water_model,
                box_shape=self.recipe.box_shape,
                box_edges=self.recipe.box_edges,
                solvent_closeness=self.recipe.solvent_closeness
            ) 
            with open(tleap_input_file, "w") as ifile:
                ifile.writelines(tleap_lines)

            solvated_complex_topology = f"{parameterised_ligand.name}_complex_solv.prmtop"
            solvated_complex_coordinates = f"{parameterised_ligand.name}_complex_solv.inpcrd"
            
        else:
            tleap_input_file = mcpbpy_tleap_file
            tleap_output_file = os.path.join(directory, f"tleap_solvate.out")
            tleap_lines = _edit_mcpbpy_tleap_input(tleap_input_file) #TODO disulfide bridges?

            saveline = [line for line in tleap_lines if "saveamberparm" in line and "solv" in line][0]
            components = saveline.split()
            solvated_complex_topology = components[2]
            solvated_complex_coordinates = components[3]

            with open(tleap_input_file, "w") as ifile:
                ifile.writelines(tleap_lines)
        
        workdir = os.getcwd()
        os.chdir(directory)
        tleap_command = f"tleap -s -f {tleap_input_file} > {tleap_output_file}"
        log.info(f"Running tleap with command:")
        log.info(tleap_command)
        os.system(tleap_command)

        try:
            solvated_topology = os.path.join(
                directory, 
                solvated_complex_topology
            )
            solvated_coordinates = os.path.join(
                directory, 
                solvated_complex_coordinates
            )
            solvated_meze = dataclasses.replace(
                self, 
                topology=solvated_topology, 
                coordinates=solvated_coordinates, 
                ligand=parameterised_ligand,
                non_standard_residues=parameterised_non_standard_residues
            )
        
        except FileNotFoundError:
            print("Failed to solvate meze.")
            raise

        os.chdir(workdir)
        return solvated_meze


    def prepare_mcpb_system(self,
                            directory: str | None = None, 
                            ligand_name: str = "ligand") -> Self:
        
        if directory:
            os.makedirs(directory, exist_ok=True)

        parameterisation_directory = os.path.join(directory, "01_mcpb_parameterisation")
        os.makedirs(parameterisation_directory, exist_ok=True)

        self._validate_disulfide_bridges()

        parameterised_ligand = self.ligand.parameterise(
            directory=parameterisation_directory, 
            filename="MOL"
        )

        self.prepare_metals_for_ezaff(directory=parameterisation_directory)

        parameterised_non_standard_residues = self.parameterise_non_standard_residues(
            directory=parameterisation_directory
        )

        complex = self.write_complex(
            directory=parameterisation_directory,
            ligand_name=ligand_name,
        )
        
        return dataclasses.replace(
            self,
            parameterisation_directory=parameterisation_directory,
            topology=complex["topology"],
            coordinates=complex["coordinates"],
            ligand=parameterised_ligand,
            non_standard_residues=parameterised_non_standard_residues,
        )

    def write_mcpb_input_file(self,
                              directory: str,
                              original_pdb: str,
                              parameterised_ligand: Ligand,
                              parameterised_non_standard_residues: Optional[list[Ligand]],
                              metals: list[str],
                              ligand_name: str = "ligand") -> str:
        
        mcpb_input_file = os.path.join(directory, "mcpbpy.in")

        if "gaff" in self.recipe.ligand_forcefield:
            gaff = self.recipe.ligand_forcefield.replace("gaff", "")
        else:
            gaff = "0"

        mcpb_input_options = {
            "original_pdb": original_pdb,
            "group_name": self.recipe.group_name + f"_{ligand_name}",
            "cut_off": self.recipe.coordination_cut_off,
            "ion_mol2files": " ".join(metals),
            "naa_mol2files": f"{parameterised_ligand.name}.mol2" if not parameterised_non_standard_residues else " ".join(
                [parameterised_ligand.name + ".mol2"] +
                [residue.name + ".mol2" for residue in parameterised_non_standard_residues]
            ),
            "frcmod_files": f"{parameterised_ligand.name}.frcmod" if not parameterised_non_standard_residues else " ".join(
                [parameterised_ligand.name + ".frcmod"] +
                [residue.name + ".frcmod" for residue in parameterised_non_standard_residues]
            ),
            "software_version": self.recipe.gaussian_version,
            "ion_ids": " ".join(str(atomid+1) for atomid in self.metal_atomids),
            "large_opt": int(self.recipe.only_optimise_hydrogens),
            "force_field": self.recipe.protein_forcefield,
            "water_model": self.recipe.water_model,
            "gaff": gaff
        }

        with open(mcpb_input_file, "w") as mcpb_file:
            for key, value in mcpb_input_options.items():
                mcpb_file.write(f"{key} {value}\n") 

        return mcpb_input_file
    
    def prepare_resp_calculation(self,
                                 ligand_name: str = "ligand",
                                 split_large_files: bool = True,
                                 sbatch_options: Optional[dict] = None,
                                 additional_lines: Optional[list[str]] = None):
        
        #TODO check prepare mcpb files exist
        if not self.parameterisation_directory:
            raise ValueError("MCPB parameterisation directory not set.")
        
        metals = []
        for i, metal in enumerate(self.metals):
            metal_mcpb_resname = f"{metal.name.upper()}{i+1}"
            metals.append(f"{metal_mcpb_resname}.mol2")

        mcpb_input_file = self.write_mcpb_input_file(
            directory=self.parameterisation_directory,
            original_pdb=self.topology,
            parameterised_ligand=self.ligand,
            parameterised_non_standard_residues=self.non_standard_residues if isinstance(self.non_standard_residues, list) else None,
            metals=metals,
            ligand_name=ligand_name
        )

        workdir = os.getcwd()
        os.chdir(self.parameterisation_directory)

        mcpb_output_file = os.path.join(
            self.parameterisation_directory, "mcpb_step1.out"
        )
        mcpb_command = f"MCPB.py -i {mcpb_input_file} -s 1 > {mcpb_output_file}"
        log.info(f"Running MCPB.py step 1 with command:\n{mcpb_command}")
        os.system(mcpb_command)

        com_files = sorted(glob.glob(f"{self.parameterisation_directory}/*.com"))

        if not com_files:
            raise ValueError(f"No Gaussian .com files found in {self.parameterisation_directory}.")
        
        if split_large_files:
            self.update_gaussian_inputs(directory=self.parameterisation_directory)
            com_files = sorted(glob.glob(f"{self.parameterisation_directory}/*.com"))
            large_opt = [f for f in com_files if "large_opt" in f][0]
            geo_opt = _write_gaussian_script(
                job_name=f"{ligand_name}-g-opt",
                gaussian_version=self.recipe.gaussian_version,
                script_name=f"{ligand_name}_slurm_g_opt.sh",
                directory=self.parameterisation_directory,
                com_file=large_opt,
                sbatch_options=sbatch_options,
                additional_lines=additional_lines
            )
            os.system(f"chmod +x {geo_opt}")

        large_mk = [f for f in com_files if "large_mk" in f][0]
        mk = _write_gaussian_script(
            job_name=f"{ligand_name}-mk",
            gaussian_version=self.recipe.gaussian_version,
            script_name=f"{ligand_name}_slurm_mk.sh",
            directory=self.parameterisation_directory,
            com_file=large_mk,
            sbatch_options=sbatch_options,
            additional_lines=additional_lines
        )
        os.system(f"chmod +x {mk}")
        os.chdir(workdir)

        return dataclasses.replace(
            self,
            mcpbpy_input_file=mcpb_input_file
        )


    def update_gaussian_inputs(self,
                               directory: str):
        
        com_files = sorted(glob.glob(f"{directory}/*.com"))
        for com_file in com_files:
            if "large_mk" in com_file:
                large_file = com_file
                with open(large_file, "r") as ilarge:
                    large_lines = ilarge.readlines()

                if "Opt" in "".join(large_lines):

                    calculation_line = [line for line in large_lines if "#" in line][0]
                    replace_index = large_lines.index(calculation_line)

                    split_parts = calculation_line.split("Opt", 1)
                    level_of_theory = split_parts[0]
                    pop_analysis = split_parts[1]
                    large_optimisation_line = level_of_theory + "Opt\n"
                    population_analysis_line = level_of_theory + "guess=read geom=checkpoint" + pop_analysis

                    clear_line = [i for i, line in enumerate(large_lines) if "CLR" in line][0]
                    header_end = clear_line + 3

                    large_opt_lines = large_lines.copy()
                    large_opt_lines[replace_index] = large_optimisation_line

                    large_mk_lines = large_lines[:header_end]
                    large_mk_lines[replace_index] = population_analysis_line

                    large_opt_file = large_file.replace("large_mk", "large_opt")
                    
                    large_opt_lines = [line.replace(line, "") if self.metal_element in line and len(line.split()) < 3 else line for line in large_opt_lines]

                    with open(large_opt_file, "w") as olarge_opt:
                        olarge_opt.writelines(large_opt_lines)

                    with open(large_file, "w") as opop:
                        opop.writelines(large_mk_lines)

        com_files = sorted(glob.glob(f"{directory}/*.com"))
        for com_file in com_files:
            with open(com_file, "r") as file:
                lines = file.readlines()
            
            new_lines = []
            for line in lines:
                if "%Mem" in line:
                    new_lines.append(f"%Mem={int(self.recipe.memory)}MB\n")
                elif "%NProcShared" in line:
                    new_lines.append(f"%NProcShared={self.recipe.nprocshared}\n")
                else:
                    new_lines.append(line)
            
            with open(com_file, "w") as file:
                file.writelines(new_lines)

    def prepare_metals_for_ezaff(self, directory: str) -> List[str]:

        metals = []
        for i, metal in enumerate(self.metals):
            metal_atomgroup = self.universe.select_atoms(f"resid {metal.resid}")
            metal_mcpb_resname = f"{metal.name.upper()}{i+1}"

            metal_atomgroup.write(f"{directory}/{metal_mcpb_resname}.pdb")

            metal_to_pdb_command = f"metalpdb2mol2.py -i {directory}/{metal_mcpb_resname}.pdb -o {directory}/{metal_mcpb_resname}.mol2 -c {self.recipe.metal_charge}"

            os.system(metal_to_pdb_command)
            metals.append(f"{metal_mcpb_resname}.mol2")

        return metals


    def write_complex(self, 
                      directory: str,
                      ligand_name: str = "ligand") -> dict[str, str]:
        
        ligand_file = os.path.join(directory, f"{self.ligand.name}.pdb")

        components = [self.coordinates, ligand_file]
        components_str = " ".join(components)

        cat_command = "cat " + components_str + f" > {directory}/{ligand_name}_complex.pdb"
        pdb4amber_command = f"pdb4amber -i {directory}/{ligand_name}_complex.pdb -o {directory}/{self.recipe.group_name}_{ligand_name}.amber.pdb"

        log.info(f"Combining complex files with command:\n{cat_command}")
        os.system(cat_command)

        log.info(f"Running pdb4amber with command:\n{pdb4amber_command}")
        os.system(pdb4amber_command)

        return {"coordinates": f"{directory}/{self.recipe.group_name}_{ligand_name}.amber.pdb",
                "topology": f"{directory}/{self.recipe.group_name}_{ligand_name}.amber.pdb"}


    def build_empirical_bonds(self):

        mcpbpy_input_file = self.mcpbpy_input_file

        self._remove_ligand_bond()
        self.restraint_file = self._remove_double_oxygen_bond()        
        
        workdir = os.getcwd()
        os.chdir(self.parameterisation_directory)
        step_2e_output_file = os.path.join(
            self.parameterisation_directory, "mcpb_step2e.out"
        )
        step_2e_command = f"MCPB.py -i {mcpbpy_input_file} -s 2e > {step_2e_output_file}"
        log.info(f"Running MCPB.py step 2e with command:\n{step_2e_command}")
        os.system(step_2e_command)
        os.chdir(workdir)
        return dataclasses.replace(
            self,
            restraint_file=self.restraint_file
        )


    def _remove_double_oxygen_bond(self):

        standard_fingerprint_file = glob.glob(
            f"{self.parameterisation_directory}/*standard.fingerprint"
        )
        if len(standard_fingerprint_file) == 0:
            raise FileNotFoundError(
                "Cannot find standard fingerprint file: "
                f"{self.parameterisation_directory}/*standard.fingerprint"
            )

        standard_fingerprint = standard_fingerprint_file[0]

        if not os.path.isfile(standard_fingerprint + "_unedited"):
            shutil.copy(
                standard_fingerprint,
                standard_fingerprint + "_unedited"
            )

        with open(standard_fingerprint, "r") as ifile:
            all_lines = ifile.readlines()
        
        oxygen_ligands = {}
        for metal, ligands in self.coordinating_residues.items():
            metal_corrected = metal + 1
            oxygen_ligands[metal_corrected] = []
            for atom in ligands:
                if atom.element == "O":
                    oxygen_ligands[metal_corrected].append(atom)

        oxygen_ids = []
        metals_with_multiple_oxygens = []
        for metal, oxygens in oxygen_ligands.items():
            n_oxygens = len(oxygens)
            if n_oxygens > 1:
                for oxygen in oxygens:
                    oxygen_ids.append(oxygen.id)
                    metals_with_multiple_oxygens.append(metal)

        atom_numbers = []
        atoms = []
        for line in all_lines:
            words = line.split()
            if "->" in words and int(words[1]) in oxygen_ids:
                atom = words[0].split("-")[-1]
                atoms.append(atom)
                atom_number = words[1]
                atom_numbers.append(int(atom_number))

        new_lines = all_lines.copy()
        harmonic_restraint_ligands = []
        if atoms and atom_numbers:
            for line in all_lines:
                words = line.split()
                if "LINK" in words:
                    metal = int(words[1].split("-")[0])
                    ligand = words[-1].split("-")
                    for atom, atom_number in zip(atoms, atom_numbers):
                        if atom in ligand and str(atom_number) in ligand and metal in metals_with_multiple_oxygens:
                            ligand_line = line
                            new_lines.remove(ligand_line)
                            temp_dict = {"metal": metal, "atom_number": atom_number}
                            harmonic_restraint_ligands.append(temp_dict)

        # build harmonic restraint for deleted bond(s):
        metal_ags = [self.universe.select_atoms(f"id {item['metal']}") for item in harmonic_restraint_ligands]
        ligand_ags = [self.universe.select_atoms(f"id {item['atom_number']}") for item in harmonic_restraint_ligands]
        distances = [np.round(MDAnalysis.analysis.distances.dist(
            atom_group_1, atom_group_2
        )[-1][0], 4) for atom_group_1, atom_group_2 in zip(metal_ags, ligand_ags)]

        elements = [[self.metal_element, ligand_ag.atoms[0].element] for ligand_ag in ligand_ags]

        force_constants = [fcfit_ep_bond(distance, element) for distance, element in zip(distances, elements)]

        restraints = []
        for metal_ag, ligand_ag, force_constant in zip(metal_ags, ligand_ags, force_constants):
            temp_dict = {metal_ag.atoms[0].id: ligand_ag}
            restraints.append(self.build_distance_restraints(
                coordinating_residues=temp_dict,
                force_constant=force_constant
            ))


        restraint_lines = [_write_distance_restraints(restraint) for restraint in restraints]
        restraint_file = os.path.join(self.parameterisation_directory, "double_oxygen_restraints.RST")
        if not os.path.isfile(restraint_file):
            with open(restraint_file, "w") as file:
                for lines in restraint_lines:
                    file.writelines(lines)
        log.info(
            f"Added harmonic restraints for deleted bonds between metal and oxygen ligand(s) to {restraint_file}."
        )

        with open(standard_fingerprint, "w") as ofile:
            ofile.writelines(new_lines)
        
        return restraint_file



    def _remove_ligand_bond(self):

        standard_fingerprint_file = glob.glob(
            f"{self.parameterisation_directory}/*standard.fingerprint"
        )
        if len(standard_fingerprint_file) == 0:
            raise FileNotFoundError(
                "Cannot find standard fingerprint file: "
                f"{self.parameterisation_directory}/*standard.fingerprint"
            )

        standard_fingerprint = standard_fingerprint_file[0]
        if not os.path.isfile(standard_fingerprint + "_unedited"):
            shutil.copy(
                standard_fingerprint,
                standard_fingerprint + "_unedited"
            )

        with open(standard_fingerprint, "r") as ifile:
            all_lines = ifile.readlines()
        
        metal_linked_atoms = [line.split()[-1].split("-") for line in all_lines if "LINK" in line]

        ligand_linked_atoms = []
        for line in all_lines:
            if "LINK" not in line:
                atom_name = line.split()[0].split("-")[2]
                atom_number = line.split()[1]
                for link in metal_linked_atoms:
                    if atom_name in link and atom_number in link and self.ligand.residue_name in line:
                        ligand_linked_atoms.append(
                            f"{atom_number}-{atom_name}"
                        )
        
        if not ligand_linked_atoms:
            log.info(
                f"Did not find a bond between the ligand {self.ligand.residue_name} and the metal"
            )
        else:
            ligand_links = []
            for line in all_lines:
                if "LINK" in line:
                    link = line.split()[-1]
                    if link in ligand_linked_atoms:
                        ligand_links.append(line)

            new_lines = [line for line in all_lines if line not in ligand_links]

            with open(standard_fingerprint, "w") as ofile:
                ofile.writelines(new_lines)
            
            for line in ligand_links:
                log.info(
                    "Succesfully removed bond: "
                    f"{line}"
                )


    def build_resp_charges(self, 
                           fix_ligand_charge: bool = True, 
                           directory: Optional[str] = None):
        
        mcpbpy_input_file = self.mcpbpy_input_file

        mcpb_input_options = _parse_mcpbpy_input(
            mcpbpy_input_file=mcpbpy_input_file
        )
    
        log_files = _check_log_files(directory=self.parameterisation_directory)

        if not hasattr(self, "ligand_resid"):
            ligand_residue_id = self.get_ligand_resid()
        else:
            ligand_residue_id = self.ligand_resid
        
        if fix_ligand_charge:
            if not directory:
                log.warning(
                    f"parent directory not set, inferring from {self.parameterisation_directory}"
                )
                directory = str(pathlib.Path(self.parameterisation_directory).parent)
    

            parameterisation_directory = os.path.join(
                directory, "02_fixed_ligand_charge"
            )
            parameterisation_directory = parameterisation_directory
            log.info(f"Creating directory: {parameterisation_directory}")
            os.makedirs(parameterisation_directory, exist_ok=True)

            ligand_files = glob.glob(
                f"{self.parameterisation_directory}/{self.ligand.residue_name}.*"
            )
            original_pdb_file = mcpb_input_options["original_pdb"]

            large_pdb_file = glob.glob(
                f"{self.parameterisation_directory}/*_large.pdb"
            )
            large_fingerprint = glob.glob(
                f"{self.parameterisation_directory}/*_large.fingerprint"
            )
            standard_pdb = glob.glob(
                f"{self.parameterisation_directory}/*_standard.pdb"
            )

            non_standard_residue_files = [res.file[0] for res in self.non_standard_residues]
            frcmod_files = glob.glob(
                f"{self.parameterisation_directory}/*.frcmod"
            )
            standard_fingerprint_file = glob.glob(
                f"{self.parameterisation_directory}/*standard.fingerprint"
            )
            if len(standard_fingerprint_file) == 0:
                raise FileNotFoundError(
                    "Cannot find standard fingerprint file: "
                    f"{self.parameterisation_directory}/*standard.fingerprint"
                )
            
            standard_fingerprint = standard_fingerprint_file[0]
            
            original_zn_files = glob.glob(
                f"{self.parameterisation_directory}/ZN*_input.mol2"
            )

            checkpoint_files = glob.glob(f"{self.parameterisation_directory}/*.chk")

            param_files = ligand_files + non_standard_residue_files + original_zn_files + \
                          log_files + frcmod_files + checkpoint_files + large_pdb_file + \
                          large_fingerprint + standard_pdb + \
                          [original_pdb_file, standard_fingerprint, mcpbpy_input_file, self.restraint_file]
            
            new_zn_files = []
            for old_file in param_files: 
                file = os.path.basename(old_file)
                new_file = os.path.join(parameterisation_directory, file)
                shutil.copy(old_file, new_file)
                if "ZN" in file:
                    new_zn_files.append(new_file)

            for file in new_zn_files:

                new_filename = file.replace("_input", "")
                os.rename(file, new_filename)

            with open(mcpbpy_input_file, "r") as ifile:
                inputs = ifile.read()
            
            new_input_file = mcpbpy_input_file.replace(
                self.parameterisation_directory, parameterisation_directory
            )
            inputs = inputs.replace(
                self.parameterisation_directory, parameterisation_directory
            )
            with open(new_input_file, "w") as ofile:
                ofile.write(inputs)
                ofile.write("\n")
                ofile.write(f"chgfix_resids {ligand_residue_id}")

            mcpbpy_input_file = new_input_file
            step_3_output_file = os.path.join(
                parameterisation_directory, "mcpb_step3.out"
            )
            step_4_output_file = os.path.join(
                parameterisation_directory, "mcpb_step4.out"
            )
        else:            
            parameterisation_directory = self.parameterisation_directory
            ion_mol2files = mcpb_input_options["ion_mol2files"]
            if isinstance(ion_mol2files, str):
                ion_mol2files = [ion_mol2files]
            
            for mol2file in ion_mol2files:
                filepath = str(pathlib.Path(mol2file).parent)
                if filepath == "" or filepath == ".":
                    mol2file = os.path.join(
                        self.parameterisation_directory,
                        mol2file
                    )
                if not os.path.isfile(mol2file):
                    raise FileNotFoundError(
                        "mol2 file for the metal does not exist: "
                        f"{mol2file}"
                    )
                filename = pathlib.Path(mol2file).stem
                new_mol2file = mol2file.replace(filename, f"{filename}_input")
                if not os.path.isfile(new_mol2file):
                    shutil.copy(mol2file, new_mol2file)

            step_3_output_file = os.path.join(
                self.parameterisation_directory, "mcpb_step3.out"
            )
            step_4_output_file = os.path.join(
                self.parameterisation_directory, "mcpb_step4.out"
            )
            

        step_3_command = f"MCPB.py -i {mcpbpy_input_file} -s 3 > {step_3_output_file}"
        log.info(f"Running MCPB.py step 3 with command:\n{step_3_command}")
        workdir = os.getcwd()
        os.chdir(parameterisation_directory)
        os.system(step_3_command)

        step_4_command = f"MCPB.py -i {mcpbpy_input_file} -s 4 > {step_4_output_file}"
        log.info(f"Running MCPB.py step 4 with command:\n{step_4_command}")
        os.system(step_4_command)
        os.chdir(workdir)

        tleap_file = glob.glob(f"{parameterisation_directory}/*tleap.in")[0]
        if not tleap_file:
            raise RuntimeError(
                "No tleap input file found after MCPB.py step 4. "
                f"Check log file: {step_4_output_file}"
            )
        
        new_coordinates = glob.glob(f"{parameterisation_directory}/*_mcpbpy.pdb")
        if not new_coordinates:
            raise RuntimeError(
                "No MCPB.py output pdb file found."
                "Check step 3 or 4 log files: "
                f"Step 3: {step_3_output_file}"
                f"Step 4: {step_4_output_file}"
            )
        else:
            new_coordinates = new_coordinates[0]
        
        new_ligand_file = glob.glob(
            f"{parameterisation_directory}/{self.ligand.residue_name[0] + self.ligand.residue_name[-1]}*.mol2"
        )[0]

        new_ligand_resname = pathlib.Path(new_ligand_file).stem
        new_ligand = Ligand(new_ligand_file, 
                            charge=_get_mol2_charge(new_ligand_file),
                            parameterised=True,
                            residue_name=new_ligand_resname,
                            frcmod_file=self.ligand.frcmod_file)

        new_non_standard_files = [glob.glob(
            f"{parameterisation_directory}/{residue.residue_name[0] + residue.residue_name[-1]}*.mol2"
        )[0] for residue in self.non_standard_residues]
        non_standard_frcmod_files = [glob.glob(
            f"{parameterisation_directory}/{residue.residue_name}.frcmod"
        )[0] for residue in self.non_standard_residues]

        new_non_standard_resnames = [pathlib.Path(file).stem for file in new_non_standard_files]
        new_non_standard_charges = [_get_mol2_charge(file) for file in new_non_standard_files]
        new_non_standard_residues = [Ligand(
            file=mol2, 
            charge=charge, 
            parameterised=True,
            residue_name=name,
            frcmod_file=frcmod
        ) for mol2, charge, name, frcmod in zip(
            new_non_standard_files, new_non_standard_charges, new_non_standard_resnames, non_standard_frcmod_files
        )]

        return dataclasses.replace(
            self,
            tleap_input_file=tleap_file,
            parameterisation_directory=parameterisation_directory,
            mcpbpy_input_file=mcpbpy_input_file,
            coordinates=new_coordinates,
            topology=new_coordinates,
            ligand=new_ligand,
            non_standard_residues=new_non_standard_residues
        )

    def build_averaged_charges(self):



        pass


@dataclass
class ColdMeze(Meze):
    recipe: ColdMezeRecipe
    exclude_resids: Optional[Union[int, list[int]]] = field(default_factory=list)
    ligand_resname: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.exclude_resids, int):
            self.exclude_resids = [self.exclude_resids]
        self.exclude_resids = set(self.exclude_resids or [])

    @classmethod
    def from_files(
        cls, 
        pdb_file: Optional[str] = None,
        topology: Optional[str] = None, 
        coordinates: Optional[str] = None, 
        exclude_resids: Optional[Union[int, list[int]]] = [],
        recipe: Optional[Union[dict, "ColdMezeRecipe"]] = None,
        ligand: Optional[Ligand] = None,
        disulfide_bridges: Optional[List[dict[str, int]]] = None,
        non_standard_residues: Optional[dict[dict]] = None,
        parameterisation_directory: Optional[str] = None,
        mcpbpy_input_file: Optional[str] = None,
        tleap_input_file: Optional[str] = None,
        ligand_resname: Optional[str] = None,
        **kwargs
    ) -> "ColdMeze":
        """
        Build a ColdMeze object from topology and coordinates.
        Passes extra kwargs into ColdMezeRecipe.
        """

        if pdb_file:
            topology = pdb_file
            coordinates = pdb_file
        if not topology or not coordinates:
            raise ValueError(
                "You must supply either a pdb file or both a topology and coordinate file."
            )
        
        if recipe is None:
            recipe = ColdMezeRecipe(**kwargs)
        elif isinstance(recipe, dict):
            recipe = ColdMezeRecipe(**recipe)
        elif not isinstance(recipe, ColdMezeRecipe):
            raise TypeError(
                f"Expected 'recipe' to be a ColdMezeRecipe, dict, or None, but got {type(recipe).__name__}"
            )

        return cls(
            topology=topology, 
            coordinates=coordinates, 
            exclude_resids=exclude_resids,
            recipe=recipe,
            ligand=ligand,
            disulfide_bridges=disulfide_bridges,
            non_standard_residues=non_standard_residues,
            parameterisation_directory=parameterisation_directory,
            mcpbpy_input_file=mcpbpy_input_file,
            tleap_input_file=tleap_input_file,
            ligand_resname=ligand_resname
        )

    def _build_restraint_mask(
            self, 
            position_restraints: str, 
            exclude_resids: Optional[Union[int, list[int]]] = [],
            additional_restraints: Optional[dict[str, Any]] = None
    ) -> str | None:
        """Build an amber-compatible restraint mask

        Args:
            position_restraints (str): what type of position restraints to apply
            additional_restraints (Optional[dict[str, Any]]): Additional restraints to apply

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
            atom.resid for atom in coordinating_atomgroups
            if atom.resid not in exclude_resids
        ]
        additional_resids = []
        if additional_restraints:
            if not {"resids"} <= additional_restraints.keys() and not {"resnames"} <= additional_restraints.keys():
                raise ValueError(
                    "additional_restraints must contain 'resids' or 'resnames' keys."
                )
            additional_resids = additional_restraints.get("resids", [])
            if isinstance(additional_resids, int):
                additional_resids = [additional_resids]
            additional_resids = set(additional_resids)

            additional_resnames = additional_restraints.get("resnames", [])
            if isinstance(additional_resnames, str):
                additional_resnames = [additional_resnames]
            additional_resnames = set(additional_resnames)
            for resname in additional_resnames:
                resname_resids = set(
                    atom.resid for atom in self.universe.select_atoms(f"resname {resname}")
                )
                additional_resids.update(resname_resids)
            additional_resids = list(additional_resids)
        if position_restraints == "solute":
            protein_resids = [atom.resid for atom in self.universe.select_atoms("protein")]
            constraint_resids = protein_resids + coordinating_resids + self.metal_resids.tolist() + additional_resids
            return f"':{_residue_restraint_mask(constraint_resids)}'"
        elif position_restraints == "backbone":
            constraint_resids = coordinating_resids + self.metal_resids.tolist() + additional_resids
            return f"'(@N,CA,C,O & !:WAT)|:{_residue_restraint_mask(constraint_resids)}'"
        elif position_restraints == "metal-coordination":
            constraint_resids = coordinating_resids + self.metal_resids.tolist() + additional_resids
            return f"':{_residue_restraint_mask(constraint_resids)}'"
        elif position_restraints is None and additional_resids:
            return f"':{_residue_restraint_mask(additional_resids)}'"
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
            engine_executable: Optional["str"] = None,
            additional_restraints: Optional[dict[str, Any]] = None
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
            path_to_engine=engine_executable or self.recipe.path_to_engine,
            model=self.recipe.model
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
            config_options["restraintmask"] = self._build_restraint_mask(
                position_restraints=position_restraints, 
                additional_restraints=additional_restraints
            )
        
        if self.recipe.model == 0:
            config_options["nmropt"] = 1

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
            engine_executable: Optional[str] = None,
            additional_restraints: Optional[dict[str, Any]] = None
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
            engine_executable=engine_executable,
            additional_restraints=additional_restraints
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
            engine_executable: Optional[str] = None,
            additional_restraints: Optional[dict[str, Any]] = None
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
            engine_executable=engine_executable,
            additional_restraints=additional_restraints
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
            engine_executable: Optional[str] = None,
            additional_restraints: Optional[dict[str, Any]] = None
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
            engine_executable=engine_executable,
            additional_restraints=additional_restraints
        )

@dataclass
class HotMeze(Meze):
    recipe: HotMezeRecipe
    restraint_file: Optional[str] = None
    ligand_resname: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()

        if self.restraint_file:
            if not os.path.isfile(self.restraint_file):
                raise FileNotFoundError(
                    f"Restraint file not found: {self.restraint_file}"
                )
        elif not self.restraint_file and self.recipe.model == 0:
            log.warning(
                "No restraint file supplied while model is 0."
                "Restraints will be determined from input files."
            )

    @classmethod
    def from_files(
        cls, 
        restraint_file: Optional[str] = "",
        recipe: Optional[Union[dict, "HotMezeRecipe"]] = None,
        pdb_file: Optional[str] = None,
        topology: Optional[str] = None, 
        coordinates: Optional[str] = None, 
        exclude_resids: Optional[Union[int, list[int]]] = [],
        ligand: Optional[Ligand] = None,
        disulfide_bridges: Optional[List[dict[str, int]]] = None,
        non_standard_residues: Optional[dict[dict]] = None,
        parameterisation_directory: Optional[str] = None,
        mcpbpy_input_file: Optional[str] = None,
        tleap_input_file: Optional[str] = None,
        ligand_resname: Optional[str] = None,
        **kwargs
    ) -> "HotMeze":
        """
        Build a HotMeze object from topology and coordinates.
        Passes extra kwargs into HotMezeRecipe.
        """
        if pdb_file:
            topology = pdb_file
            coordinates = pdb_file
        if not topology or not coordinates:
            raise ValueError(
                "You must supply either a pdb file or both a topology and coordinate file."
            )
        
        if recipe is None:
            recipe = HotMezeRecipe(**kwargs)
        elif isinstance(recipe, dict):
            recipe = HotMezeRecipe(**recipe)
        elif not isinstance(recipe, HotMezeRecipe):
            raise TypeError(
                f"Expected 'recipe' to be a HotMezeRecipe, dict, or None, but got {type(recipe).__name__}"
        )

        return cls(
            topology=topology, 
            coordinates=coordinates, 
            recipe=recipe,
            restraint_file=restraint_file,
            ligand=ligand,
            disulfide_bridges=disulfide_bridges,
            non_standard_residues=non_standard_residues,
            parameterisation_directory=parameterisation_directory,
            mcpbpy_input_file=mcpbpy_input_file,
            tleap_input_file=tleap_input_file,
            exclude_resids=exclude_resids,
            ligand_resname=ligand_resname
        )

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
            write_frequency: Optional[int] = 100000,
            distance_write_frequency: Optional[int] = 10000,
            additional_restraints: Optional[dict[str, Any]] = None
    ):
        recipe = HotMezeRecipe(
            workdir=workdir or self.recipe.workdir,
            nb_cutoff=nb_cutoff or self.recipe.nb_cutoff,
            runtime= runtime or self.recipe.runtime,
            dt=timestep or self.recipe.dt,
            temperature=temperature or self.recipe.temperature,
            pressure=pressure or self.recipe.pressure,
            path_to_engine=engine_executable or self.recipe.path_to_engine,
            model=self.recipe.model
        )

        config_options = {"cut": recipe.nb_cutoff,
                          "ntpr": write_frequency,
                          "ntwx": write_frequency,
                          "ntwr": write_frequency,
                          "irest": 1,
                          "ntx": 5, 
                          "iwrap": 0}

        if self.recipe.model == 0:
            config_options["nmropt"] = 1

        protocol = bss.Protocol.Production(
            timestep=bss.Types.Time(recipe.dt, "ps"),
            runtime=bss.Types.Time(recipe.runtime, "ns"),
            temperature=bss.Types.Temperature(recipe.temperature, "K"),
            pressure=bss.Types.Pressure(recipe.pressure, "atm")
        )
        if os.path.isfile(self.restraint_file):
            step_restraint_file = os.path.join(recipe.workdir, "restraints.RST")
            shutil.copyfile(self.restraint_file, step_restraint_file)

        return super()._run(
            protocol=protocol,
            recipe=recipe,
            system=system,
            process_name=process_name,
            config_options=config_options,
            distance_write_frequency=distance_write_frequency,
            additional_restraints=additional_restraints
        )
        
@dataclass
class QuantumMeze(Meze):
    recipe: MezeRecipe
    exclude_resids: Optional[Union[int, list[int]]] = field(default_factory=list)
    metal_resids_for_distance_restraints: Optional[Union[int, list[int]]] = None

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.exclude_resids, int):
            self.exclude_resids = [self.exclude_resids]
        
        if isinstance(self.metal_resids_for_distance_restraints, int):
            self.metal_resids_for_distance_restraints = [self.metal_resids_for_distance_restraints]
        
        self.exclude_resids = set(self.exclude_resids or [])
        self.qm_region = self._define_qm_region()
        self.qm_charge = self._get_qm_charge()
        self.distance_restraints = self._prepare_distance_restraints(
            self.metal_resids_for_distance_restraints
        )

    @classmethod
    def from_files(
        cls, 
        topology: str, 
        coordinates: str, 
        exclude_resids: Optional[Union[int, list[int]]] = None,
        metal_resids_for_distance_restraints: Optional[Union[int, list[int]]] = None,
        **kwargs
    ) -> "QuantumMeze":
        """
        Build a Meze object from topology and coordinates.
        Passes extra kwargs into MezeRecipe.
        """
        recipe = MezeRecipe(**kwargs)
        return cls(
            topology=topology, 
            coordinates=coordinates, 
            exclude_resids=exclude_resids,
            metal_resids_for_distance_restraints=metal_resids_for_distance_restraints,
            recipe=recipe
        )

    def _define_qm_region(
            self, 
            resids_to_exclude: Optional[Union[int, list[int]]] = None
    ) -> dict[str, list]:
        """Get a simple QM region 

        Returns:
            dict[str, list]: QM region split into a list of whole residues and atom ids
        """
        exclude_resids = set(self.exclude_resids or [])
        resids_to_exclude = resids_to_exclude or self.metal_resids_for_distance_restraints
        if resids_to_exclude is not None:
            if isinstance(resids_to_exclude, int):
                resids_to_exclude = [resids_to_exclude]
            exclude_resids.update(resids_to_exclude)

        excluded_atoms = set()
        for metal_atom_idx, metal_ligands in self.coordinating_residues.items():
            metal_resid = self.universe.select_atoms(f"id {metal_atom_idx}").resids[0]
            if metal_resid in exclude_resids:
                excluded_atoms.add(metal_atom_idx)
                for residue in metal_ligands.residues:
                    exclude_resids.add(residue.resid)

        qm_region_atom_ids = set()
        qm_region_whole_residues = set()

        protein = self.universe.select_atoms("protein")

        for metal_id, metal_ligands in self.coordinating_residues.items():
            if metal_id in excluded_atoms:
                continue 

            metal_atom = self.universe.select_atoms(f"id {metal_id}")[0]
            if metal_atom.resid not in exclude_resids:
                qm_region_atom_ids.add(str(metal_id))

            for residue in metal_ligands.residues:
                if residue.resid in exclude_resids:
                    continue  

                if residue in protein.residues:
                    side_chain_atoms = self._get_side_chain_selection(residue)
                    qm_region_atom_ids.add(side_chain_atoms)
                else:
                    qm_region_whole_residues.add(residue.resid)

        qm_region = {
            "whole_residues": list(qm_region_whole_residues),
            "atom_ids": list(qm_region_atom_ids),
        }

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
    
    def _get_qm_charge(self) -> int:

        charge = 0.0
        for residue in self.qm_region["whole_residues"]:
            atoms = self.universe.select_atoms(f"resid {residue}")
            charge += atoms.charges.sum()

        for atom_selection in self.qm_region["atom_ids"]:
            atom_id = atom_selection.replace("-", " to ")
            atoms = self.universe.select_atoms(f"id {atom_id}")
            charge += atoms.charges.sum()
        return int(np.round(charge))

    
    def _write_qm_namelist(self, qm_theory: str = "DFTB3"):

        parsed_whole_residues = _residue_restraint_mask(self.qm_region["whole_residues"])
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
    
    def _prepare_distance_restraints(
        self,
        resids_for_distance_restraints: Optional[Union[int, list[int]]] = None
    ) -> Optional[list[str]]:
        if not resids_for_distance_restraints:
            return None
        
        if isinstance(resids_for_distance_restraints, int):
            resids_for_distance_restraints = [resids_for_distance_restraints]

        atom_ids = [
            self.universe.select_atoms(f"resid {resid}").ids[0]
            for resid in resids_for_distance_restraints
        ]
        distance_restraints_dict = self.build_distance_restraints(atom_ids)
        return _write_distance_restraints(distance_restraints_dict)
    
    def run_qm(
        self,
        recipe: MezeRecipe,
        protocol: bss.Protocol,
        system: Optional[bssSystem] = None,
        process_name: Optional[str] = "qm-meze-run",
        config_options: Optional[dict] = None,
        qm_theory: str = "DFTB3",
        metal_resids_for_distance_restraints: Optional[Union[int, list[int]]] = None,
        is_gpu: bool = False,
        additional_restraints: Optional[dict[str, Any]] = None
    ) -> "QuantumMeze":
        
        config_options["ntc"] = 1
        config_options["ntf"] = 1

        if not additional_restraints:
            disres=metal_resids_for_distance_restraints or self.metal_resids_for_distance_restraints
        else: 
            if not {"resids"} <= additional_restraints.keys() and not {"resnames"} <= additional_restraints.keys():
                raise ValueError(
                    "additional_restraints must contain 'resids' or 'resnames' keys."
                )
            additional_resids = additional_restraints.get("resids", [])
            if isinstance(additional_resids, int):
                additional_resids = [additional_resids]
            additional_resids = set(additional_resids)

            additional_resnames = additional_restraints.get("resnames", [])
            if isinstance(additional_resnames, str):
                additional_resnames = [additional_resnames]
            additional_resnames = set(additional_resnames)
            for resname in additional_resnames:
                resname_resids = set(
                    atom.resid for atom in self.universe.select_atoms(f"resname {resname}")
                )
                additional_resids.update(resname_resids)
            additional_resids = list(additional_resids)
            disres = additional_resids
            
        self.qm_region = self._define_qm_region(
            resids_to_exclude=metal_resids_for_distance_restraints
        )
        
        qm_namelist = self._write_qm_namelist(qm_theory=qm_theory)

        if disres is not None:
            distance_restraints = self._prepare_distance_restraints(disres)
        else:
            distance_restraints = self.distance_restraints

        if distance_restraints:
            config_options["nmropt"] = 1
            restraint_namelist = ["&wt TYPE='DUMPFREQ', istep1=1 /"]
        else:
            restraint_namelist = []
        
        namelist = qm_namelist + restraint_namelist 

        return super()._run(
            protocol=protocol,
            recipe=recipe,
            system=system,
            process_name=process_name,
            config_options=config_options,
            namelist_options=namelist,
            is_gpu=is_gpu,
            distance_restraints=distance_restraints
        )

@dataclass
class ColdQuantumMeze(QuantumMeze):
    recipe: ColdMezeRecipe
    exclude_resids: Optional[Union[int, list[int]]] = field(default_factory=list)
    metal_resids_for_distance_restraints: Optional[Union[int, list[int]]] = None

    @classmethod
    def from_files(
        cls, 
        topology: Optional[str] = None, 
        coordinates: Optional[str] = None, 
        exclude_resids: Optional[Union[int, list[int]]] = [],
        recipe: Optional[Union[dict, "ColdMezeRecipe"]] = None,
        disulfide_bridges: Optional[List[dict[str, int]]] = None,
        metal_resids_for_distance_restraints: Optional[Union[int, list[int]]] = None,
        **kwargs
    ) -> "ColdQuantumMeze":
        """
        Build a Meze object from topology and coordinates.
        Passes extra kwargs into MezeRecipe.
        """
        
        if recipe is None:
            recipe = ColdMezeRecipe(**kwargs)
        elif isinstance(recipe, dict):
            recipe = ColdMezeRecipe(**recipe)
        elif not isinstance(recipe, ColdMezeRecipe):
            raise TypeError(
                f"Expected 'recipe' to be a ColdMezeRecipe, dict, or None, but got {type(recipe).__name__}"
            )
        return cls(
            topology=topology, 
            coordinates=coordinates,
            exclude_resids=exclude_resids,
            metal_resids_for_distance_restraints=metal_resids_for_distance_restraints,
            disulfide_bridges=disulfide_bridges,
            recipe=recipe
        )

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
            qm_theory: Optional[str] = "DFTB3",
            metal_resids_for_distance_restraints: Optional[Union[int, list[int]]] = None,
            additional_restraints: Optional[dict[str, Any]] = None

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
        return super().run_qm(
            protocol=protocol,
            recipe=recipe,
            system=system,
            process_name=process_name,
            qm_theory=qm_theory,
            metal_resids_for_distance_restraints=metal_resids_for_distance_restraints,
            config_options=config_options,
            is_gpu=False,
            additional_restraints=additional_restraints
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
            engine_executable: Optional[str] = None,
            qm_theory: Optional[str] = "DFTB3",
            metal_resids_for_distance_restraints: Optional[Union[int, list[int]]] = None,
            additional_restraints: Optional[dict[str, Any]] = None
    ) -> "ColdQuantumMeze":  
        disres=metal_resids_for_distance_restraints or self.metal_resids_for_distance_restraints

        return self.run(
            protocol_type="minimisation",
            system=system,
            workdir=workdir,
            process_name=process_name,
            max_cycles=max_cycles,
            n_sd_cycles=n_sd_cycles,
            nb_cutoff=nb_cutoff,
            method=method,
            engine_executable=engine_executable,
            qm_theory=qm_theory,
            metal_resids_for_distance_restraints=disres,
            additional_restraints=additional_restraints
        )
    
    def heat(
            self,
            system: Optional[bssSystem] = None,
            workdir: Optional[str] = None,
            restart: Optional[bool] = False,
            timestep: Optional[Union[float, bssTemperature]] = 0.001,
            runtime: Optional[Union[float, bssTime]] = None,
            temperature: Optional[Union[float, bssTemperature]] = None,
            start_temperature: Optional[Union[float, bssTemperature]] = 300,
            end_temperature: Optional[Union[float, bssTemperature]] = 300,
            process_name: Optional[str] = "qm-nvt",
            engine_executable: Optional[str] = None,
            qm_theory: Optional[str] = "DFTB3",
            metal_resids_for_distance_restraints: Optional[Union[int, list[int]]] = None,
            additional_restraints: Optional[dict[str, Any]] = None
    ) -> "ColdQuantumMeze":
        disres=metal_resids_for_distance_restraints or self.metal_resids_for_distance_restraints
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
            engine_executable=engine_executable,
            qm_theory=qm_theory,
            metal_resids_for_distance_restraints=disres,
            additional_restraints=additional_restraints
        )
    
@dataclass
class HotQuantumMeze(QuantumMeze):
    recipe: HotMezeRecipe
    exclude_resids: Optional[Union[int, list[int]]] = field(default_factory=list)
    metal_resids_for_distance_restraints: Optional[Union[int, list[int]]] = None

    @classmethod
    def from_files(
        cls, 
        topology: Optional[str] = None, 
        coordinates: Optional[str] = None, 
        exclude_resids: Optional[Union[int, list[int]]] = [],
        recipe: Optional[Union[dict, "ColdMezeRecipe"]] = None,
        disulfide_bridges: Optional[List[dict[str, int]]] = None,
        metal_resids_for_distance_restraints: Optional[Union[int, list[int]]] = None,
        **kwargs
    ) -> "HotQuantumMeze":
        """
        Build a Meze object from topology and coordinates.
        Passes extra kwargs into MezeRecipe.
        """

        if recipe is None:
            recipe = HotMezeRecipe(**kwargs)
        elif isinstance(recipe, dict):
            recipe = HotMezeRecipe(**recipe)
        elif not isinstance(recipe, HotMezeRecipe):
            raise TypeError(
                f"Expected 'recipe' to be a HotMezeRecipe, dict, or None, but got {type(recipe).__name__}"
            )
        return cls(
            topology=topology, 
            coordinates=coordinates, 
            exclude_resids=exclude_resids,
            metal_resids_for_distance_restraints=metal_resids_for_distance_restraints,
            recipe=recipe,
            disulfide_bridges=disulfide_bridges
        )
    
    def run(
            self,
            workdir: Optional[str],
            system: Optional[bssSystem] = None,
            process_name: Optional[str] = "qm-meze-run",
            nb_cutoff: Optional[float] = None,
            timestep: Optional[Union[float, bssTime]] = 0.001,
            runtime: Optional[Union[float, bssTime]] = None,
            temperature: Optional[Union[float, bssTemperature]] = 300,
            pressure: Optional[Union[float, bssPressure]] = None,
            engine_executable: Optional[str] = None,
            write_frequency: Optional[int] = 500,
            qm_theory: Optional[str] = "DFTB3",
            metal_resids_for_distance_restraints: Optional[Union[int, list[int]]] = None,
            additional_restraints: Optional[dict[str, Any]] = None
    ) -> "HotQuantumMeze":
        disres=metal_resids_for_distance_restraints or self.metal_resids_for_distance_restraints
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
        
        protocol = bss.Protocol.Production(
            timestep=bss.Types.Time(recipe.dt, "ps"),
            runtime=bss.Types.Time(recipe.runtime, "ps"),
            temperature=bss.Types.Temperature(recipe.temperature, "K"),
            pressure=None
        )

        return super().run_qm(
            protocol=protocol,
            recipe=recipe,
            system=system,
            process_name=process_name,
            qm_theory=qm_theory,
            metal_resids_for_distance_restraints=disres,
            config_options=config_options,
            is_gpu=False,
            additional_restraints=additional_restraints
        )
