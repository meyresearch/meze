from dataclasses import (
    dataclass,
    field
)
import dataclasses
import glob
import json
import warnings
import logging
import numpy as np
from pydantic import (
    Field,
    field_validator,
    BaseModel,
    ConfigDict
)
from typing import (
    Any,
    List,
    Optional,
    Literal,
    Union,
    Self,
    TYPE_CHECKING
)
import pandas as pd
import pickle
import pathlib
from .ligand import Ligand
import os
import MDAnalysis as mda
import MDAnalysis.analysis.distances
from MDAnalysis.topology.guessers import guess_types
from MDAnalysis.core.groups import Residue as mdaResidue
import BioSimSpace as bss
from BioSimSpace._SireWrappers import System as bssSystem
if TYPE_CHECKING:
    from BioSimSpace.Protocol._protocol import Protocol as bssProtocol
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
    _edit_mcpbpy_tleap_input,
    pdb_to_sdf,
    _set_n_somd_moves,
    _remove_gpu_from_fep_configs,
    _write_somd_restraints
)
from .helpers import _check_ambertools
import shutil
from rich.logging import RichHandler
from rich.console import Console
import subprocess
import csv
import multiprocessing
if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method("fork")
warnings.filterwarnings(
    "ignore", message="to-Python converter for std::__1::vector"
)
logging.getLogger("numexpr.utils").setLevel(logging.ERROR)
logging.getLogger("MDAnalysis").setLevel(logging.ERROR)
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
    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_default=True
    )
    workdir: str = Field(
        default_factory=os.getcwd, description="Working directory"
    )
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
    memory: float = Field(
        12_000, description="Memory for Gaussian calculations in MB"
    )
    nprocshared: int = Field(
        8, description="Number of processors for Gaussian calculations"
    )
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
    n_repeats: int = Field(
        3, ge=1, description="Number of repeats"
    )
    temperature: Union[float, bss.Types.Temperature] = Field(
        300.0,  description="Simulation temperature in kelvin"
    )
    pressure: Union[float, bss.Types.Pressure] = Field(
        1.0, description="Simulation pressure in atm"
    )
    nb_cutoff: Union[float, bss.Types.Length] = Field(
        12.0, description="Cut-off for electrostatics interactions"
    )

    @field_validator("model", mode="after")
    @classmethod
    def validate_model(cls, v):
        if v is None:
            return v
        try:
            return int(v)
        except (TypeError, ValueError):
            raise ValueError(f"Cannot covert model='{v}' to int")

    @field_validator("temperature", mode="after")
    @classmethod
    def validate_temperature(cls, value):
        if isinstance(value, bss.Types.Temperature):
            return value
        value = float(value)
        if value < 0:
            raise ValueError(
                "temperature must be greater than or equal to 0 K"
            )
        return bss.Types.Temperature(value, "kelvin")

    @field_validator("pressure", mode="after")
    @classmethod
    def validate_pressure(cls, value):
        if isinstance(value, bss.Types.Pressure):
            return value
        value = float(value)
        if value < 0:
            raise ValueError(
                "pressure must be greater than or equal to 0 atm"
            )
        return bss.Types.Pressure(value, "atm")

    @field_validator("nb_cutoff", mode="after")
    @classmethod
    def validate_cutoff_distance(cls, value):
        if isinstance(value, bss.Types.Length):
            return value
        value = float(value)
        if value < 0:
            raise ValueError(
                "nb_cutoff must be greater than or equal to 0 atm"
            )
        return bss.Types.Length(value, "angstrom")

    def __str__(self) -> str:
        """Print recipe information as JSON
        """
        return self.model_dump_json(indent=4, fallback=str, warnings="none")

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __setitem__(self, key: str, value):
        setattr(self, key, value)

    def to_json(self, file: str):
        with open(file, "w") as ofile:
            ofile.write(self.model_dump_json(indent=2, fallback=str))


class ColdMezeRecipe(MezeRecipe):
    """Meze workflow recipe for minimisation and equilibration
    """
    max_cycles: int = Field(
        1000, ge=0, description="Number of minimisation cycles"
    )
    n_sd_cycles: int = Field(
        1000,
        ge=0,
        description="Number of steepest descent cycles (if min_method=1)"
    )
    min_method: int = Field(
        1,
        ge=0,
        description=(
            "Run steepest descent for n_sd_cycles, "
            "then conjugate gradient"
        )
    )
    barostat: int = Field(
        2, ge=1, le=2, description="Type of barostat, 1: Berendsen, 2: MC"
    )

    runtime: Union[float, bss.Types.Time] = Field(
        100.0, description="Simulation time in picoseconds"
    )
    dt: Union[float, bss.Types.Time] = Field(
        0.001, description="Integrator timestep, in picoseconds"
    )
    start_temperature: float = Field(
        300.0, description="Simulation start temperature in kelvin"
    )
    end_temperature: float = Field(
        300.0, description="Simulation end temperature in kelvin"
    )
    restraint_weight: float = Field(
        100.0, ge=0,
        description="Force constant for positional restraints "
        "in kcal/(mol*Å^2)"
    )

    @field_validator("dt", "runtime", mode="after")
    @classmethod
    def validate_time(cls, value):
        if isinstance(value, bss.Types.Time):
            return value
        if value <= 0:
            raise ValueError(
                "dt, time must be greater than 0 picoseconds"
            )
        return bss.Types.Time(value, "picoseconds")

    @field_validator("start_temperature", "end_temperature", mode="after")
    @classmethod
    def validate_temperature_range(cls, value):
        if isinstance(value, bss.Types.Temperature):
            return value
        value = float(value)
        if value < 0:
            raise ValueError(
                "temperature must be greater than or equal to 0 K"
            )
        return bss.Types.Temperature(value, "kelvin")


class HotMezeRecipe(MezeRecipe):
    """Meze workflow recipe for production runs
    """
    runtime: Union[float, bss.Types.Time] = Field(
        100.0, description="Simulation time in nanoseconds"
    )
    dt: Union[float, bss.Types.Time] = Field(
        0.002, description="Integrator timestep, in picoseconds"
    )

    @field_validator("runtime", mode="after")
    @classmethod
    def validate_time(cls, value):
        if isinstance(value, bss.Types.Time):
            return value
        if value <= 0:
            raise ValueError(
                "dt must be greater than 0 nanoseconds"
            )
        return bss.Types.Time(value, "nanoseconds")

    @field_validator("dt", mode="after")
    @classmethod
    def validate_timestep(cls, value):
        if isinstance(value, bss.Types.Time):
            return value
        if value <= 0:
            raise ValueError(
                "dt must be greater than 0 picoseconds"
            )
        return bss.Types.Time(value, "picoseconds")


class AlchemicalMezeRecipe(MezeRecipe):
    """Meze workflow recipe for alchemical free energy calculations
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_default=True
    )
    n_lambdas: int = Field(
        16, ge=3, description="Number of lambda windows"
    )
    sampling_time: Union[float, bss.Types.Time] = Field(
        4.0, description="Runtime for each lambda window in ns."
    )
    restart_interval: int = Field(
        500, ge=1, description="N:o steps with which restart files are written"
    )
    report_interval: int = Field(
        500, ge=1, description="N:o steps with which outputs are written"
    )
    flexible_align: bool = Field(
        False,
        description="Whether to flexibly align ligands for single topology."
    )
    ring_breaks: bool = Field(
        True, description="Whether to allow ring breaking in merging."
    )
    ring_size_changes: bool = Field(
        True, description="Whether to allow ring sizes to change for merging."
    )
    engine: str = Field(
        "SOMD", description="Which MD engine to use for the alchemistry"
    )
    dt: float = Field(
        0.002, description="Integrator timestep, in picoseconds"
    )
    minimise_lambda: bool = Field(
        True, description="Whether to carry out minimisation at each lambda"
    )
    lambda_minimisation_steps: int = Field(
        5000,
        description=(
            "Number of minimisation steps to do "
            "at each lambda minimisation."
        )
    )

    @field_validator("sampling_time", mode="after")
    @classmethod
    def validate_sampling_time(cls, value):
        if isinstance(value, bss.Types.Time):
            return value
        value = float(value)
        if value <= 0:
            raise ValueError("sampling_time must be greater than 0 ns")
        return bss.Types.Time(value, "nanoseconds")

    @field_validator("dt", mode="after")
    @classmethod
    def validate_picosecond_times(cls, value):
        if isinstance(value, bss.Types.Time):
            return value
        if value <= 0:
            raise ValueError(
                "dt must be greater than 0 picoseconds"
            )
        return bss.Types.Time(value, "picoseconds")


@dataclass
class Meze:
    topology: str
    coordinates: str
    recipe: MezeRecipe
    disulfide_bridges: Optional[List[dict[str, int]]] = None
    ligand: Optional[Ligand] = None
    ligand_resid: Optional[int] = None
    non_standard_residues: dict[dict] | List[Ligand] = field(
        default_factory=dict
    )
    parameterisation_directory: Optional[str] = None
    mcpbpy_input_file: Optional[str] = None
    tleap_input_file: Optional[str] = None
    restraint_file: Optional[str] = None
    exclude_resids: Optional[Union[int, list[int]]] = field(
        default_factory=list
    )
    ligand_resname: Optional[str] = None
    stage: str = "bound"

    def __post_init__(self):

        self._check_file_exists(self.topology)
        self._check_file_exists(self.coordinates)

        coordinate_extension = os.path.splitext(self.coordinates)[1]
        coordinate_format = self._get_coordinate_fileformat(
            coordinate_file_extension=coordinate_extension
        )
        topology_extension = os.path.splitext(self.topology)[1]

        self.universe = self._set_universe(
            coordinate_extension=coordinate_extension,
            topology_extension=topology_extension,
            coordinate_format=coordinate_format
        )

        if self.stage == "bound":
            self._set_metal()
            self.coordinating_residues = (
                self._get_metal_coordinating_residues()
            )
            self._set_protein()
        elif self.stage == "unbound":
            self.metals = None
            self.metal_resids = []
            self.metal_atomids = []
            self.coordinating_residues = {}
            self.protein = None
        else:
            message = (
                f"Unrecognised stage set: {self.stage}"
            )
            log.error(message)
            raise ValueError(message)

        self._set_waters()
        self._setup_bss_system()

        if self.non_standard_residues and isinstance(
            self.non_standard_residues, dict
        ):
            self._validate_non_standard_residues()

        if self.ligand and self.ligand.parameterised and not self.ligand_resid:
            self.ligand_resid = self.get_ligand_resid()
            self.ligand_resname = self.ligand.residue_name
        elif self.ligand and self.ligand.parameterised and self.ligand_resid:
            self.ligand_resname = self.ligand.residue_name
        elif not self.ligand and self.ligand_resname:
            log.info(
                "Inferring ligand from ligand residue name: "
                f"{self.ligand_resname}"
            )
            self._set_ligand()
            self.ligand_resid = self.get_ligand_resid()
        else:
            log.warning(
                "Ligand not set by user in meze construction. "
            )

    @staticmethod
    def _check_file_exists(file):
        if not os.path.isfile(file):
            message = f"File '{file}' not found."
            log.error(message)
            raise FileNotFoundError(message)

    @staticmethod
    def _get_coordinate_fileformat(coordinate_file_extension):
        if coordinate_file_extension in [".rst7"]:
            return "RESTRT"
        else:
            return None

    def _set_universe(
            self,
            coordinate_extension,
            topology_extension,
            coordinate_format
    ):
        if coordinate_extension == topology_extension:
            with warnings.catch_warnings(record=True):
                warnings.filterwarnings(
                    "always",
                    message=r"Unknown element.*empty element record",
                    category=UserWarning,
                    module=r"MDAnalysis\.topology\.PDBParser",
                )
                universe = mda.Universe(
                    self.topology,
                )
                guessed_elements = guess_types(universe.atoms.names)
                universe.add_TopologyAttr(
                    "elements", guessed_elements
                )
        else:
            universe = mda.Universe(
                self.topology,
                self.coordinates,
                topology_format="PARM7",
                format=coordinate_format
            )
        return universe

    def __str__(self) -> str:
        return _pretty(self)

    def save(self, filename: str) -> str:
        suffix = pathlib.Path(filename).suffix
        if not suffix:
            filename += ".pkl"
        with open(filename, "wb") as file:
            pickle.dump(self, file)
        return filename

    def add_to_sofra(
            self,
            filename: str,
            key: str,
            pickle_file: Optional[str] = None,
            extra_fields: Optional[dict] = None
    ):
        if extra_fields is not None:
            try:
                json.dumps(extra_fields)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"extra_fields contians non-JSON-serialisable values: {e}"
                )
        new_entry = {
            key: {}
        }
        if pickle_file is not None:
            new_entry[key]["pickle_file"] = pickle_file
        if self.parameterisation_directory is not None:
            new_entry[key]["parameterisation_directory"] = (
                self.parameterisation_directory
            )
        if self.mcpbpy_input_file is not None:
            new_entry[key]["mcpbpy_input_file"] = self.mcpbpy_input_file
        if self.tleap_input_file is not None:
            new_entry[key]["tleap_input_file"] = self.tleap_input_file
        if self.restraint_file is not None:
            new_entry[key]["restraint_file"] = self.restraint_file
        if self.topology is not None:
            new_entry[key]["topology"] = self.topology
        if self.coordinates is not None:
            new_entry[key]["coordinates"] = self.coordinates

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
        disulfide_bridges: Optional[List[dict[str, int]]] = None,
        ligand: Optional[Ligand] = None,
        ligand_resid: Optional[int] = None,
        non_standard_residues: dict[dict] | List[Ligand] = None,
        parameterisation_directory: Optional[str] = None,
        mcpbpy_input_file: Optional[str] = None,
        tleap_input_file: Optional[str] = None,
        restraint_file: Optional[str] = None,
        exclude_resids: Optional[Union[int, list[int]]] = None,
        ligand_resname: Optional[str] = None,
        stage: Optional[str] = "bound",
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
            disulfide_bridges=disulfide_bridges,
            ligand=ligand,
            ligand_resid=ligand_resid,
            non_standard_residues=non_standard_residues,
            parameterisation_directory=parameterisation_directory,
            mcpbpy_input_file=mcpbpy_input_file,
            tleap_input_file=tleap_input_file,
            restraint_file=restraint_file,
            exclude_resids=exclude_resids,
            ligand_resname=ligand_resname,
            recipe=recipe,
            stage=stage
        )

    def _set_waters(self):
        self.crystal_waters = self.universe.select_atoms("water")

    def _set_ligand(self):

        ag = self.universe.select_atoms(f"resname {self.ligand_resname}")
        if len(ag) == 0:
            log.warning(
                f"Could not find ligand with resname {self.ligand_resname}"
                "Ligand not set for system. "
                "Consider adding a ligand with meze.add_ligand()"
                "or use a pickle file to load in a meze object"
            )

        self.ligand = Ligand(
            file=[self.coordinates, self.topology],
            name=self.ligand_resname,
            charge=ag.charges.sum(),
            parameterised=True,
            residue_name=self.ligand_resname
        )

    def get_mutatable_ligand_molecule(self):

        if not self.ligand_resname:
            raise RuntimeError(
                "Ligand residue name not set. "
                "Cannot determine mutatable ligand."
            )

        residues = self.system.getResidues()
        if len(residues) == 0:
            raise RuntimeError(
                "No residues found in BioSimSpace system for meze with file:"
                f"\n{self.coordinates}"
            )
        ligand_residues = [
            residue for residue in residues if residue.name().upper() == (
                self.ligand_resname
            )
        ]
        if not ligand_residues:
            raise RuntimeError(
                "Could not find any ligand residues for meze object with file:"
                f"\n{self.coordinates}"
            )
        if len(ligand_residues) > 1:
            message = "Cannot extract ligand with multiple residues"
            log.error(message)
            raise NotImplementedError(message)
        return ligand_residues[0].toMolecule()

    def get_small_molecule_resname(self) -> str | None:  # DEPRECATED

        selection = self.universe.select_atoms(
            "not protein and not water"
        )
        non_standard_residues = [
            "MOH", "DOH", "Na+",
            "CL-", "ASZ", "GLZ",
            "HDZ", "HEZ", "CYZ", "CYM",
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

        metal_atom_ids = metal_atom_ids or list(
            self.coordinating_residues.keys()
        )
        ligand_residues = coordinating_residues or self.coordinating_residues
        if coordinating_residues is not None:
            if not isinstance(coordinating_residues, dict):
                raise TypeError(
                    "coordinating_residues must be a dict, got "
                    f"{type(coordinating_residues).__name__}"
                )
            if not all(isinstance(k, int) for k in coordinating_residues):
                raise TypeError(
                    f"coordinating_residues keys must be ints (atom IDs), "
                    f"got {set(
                        type(k).__name__ for k in coordinating_residues
                    )}"
                )
            if not all(
                isinstance(
                    v, mda.AtomGroup
                ) for v in coordinating_residues.values()
            ):
                raise TypeError(
                    "coordinating_residues values must be "
                    "MDAnalysis AtomGroups"
                )
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

                    if self.ligand and ligating_atom.resname.upper() == (
                        self.ligand.residue_name
                    ):
                        continue
                    key = (metal_id, ligating_atom.id)
                    atom_group_2 = self.universe.select_atoms(
                        f"resid {ligating_atom.resid} and "
                        f"name {ligating_atom.name}"
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

        metal_atom_ids = list(self.metal_atomids)
        distance_restraints_dict = self.build_distance_restraints(
            metal_atom_ids
        )
        return _write_distance_restraints(distance_restraints_dict)

    def write_restrained_atoms_pdb(
        self,
        output_path: str,
        restraints: Optional[
            dict[tuple[int, int], tuple[float, float, float]] | list[str]
        ] = None
    ) -> None:
        """Write a PDB file containing all atoms involved in
           distance restraints.

        Args:
            output_path (str): Path for the output PDB file.
            restraints (dict, optional): Either a dict from
            build_distance_restraints(), a list of RST format strings,
            or None (calls build_distance_restraints() by default)
        """
        if restraints is None:
            restraints = self.build_distance_restraints()

        atom_ids: list[int] = []
        if isinstance(restraints, dict):
            for metal_id, ligating_atom_id in restraints.keys():
                atom_ids.append(metal_id)
                atom_ids.append(ligating_atom_id)
        else:
            for line in restraints:
                if "iat=" in line:
                    after_iat = line.split("iat=")[1]
                    raw_indices = after_iat.split(",")[:2]
                    atom_ids.extend(int(iat) for iat in raw_indices)

        id_selection = " or ".join(f"id {aid}" for aid in sorted(atom_ids))
        restrained_atoms = self.universe.select_atoms(id_selection)
        restrained_atoms.write(output_path)

    def _prepare_angle_restraints(
        self
    ) -> Optional[list[str]]:

        metal_atom_ids = list(self.metal_atomids)
        angle_restraints_dict = self.build_angle_restraints(metal_atom_ids)
        return _write_distance_restraints(angle_restraints_dict)

    def build_angle_restraints(
            self,
            metal_atom_ids: Optional[list[int]] = None,
            coordinating_residues: Optional[dict[int, mda.AtomGroup]] = None,
            force_constant: Optional[float] = 100.0,
            flat_bottom_radius: Optional[float] = 1.00,
            exclude_residues: Optional[Union[int, list[int]]] = None
    ) -> dict[tuple[int, int], tuple[float, float, float]]:
        """
        Enforce "angle" restraints through
        additional distance restraints between vertex atoms.
        """
        metal_atom_ids = metal_atom_ids or list(
            self.coordinating_residues.keys()
        )
        ligand_residues = coordinating_residues or self.coordinating_residues
        if coordinating_residues is not None:
            if not isinstance(coordinating_residues, dict):
                raise TypeError(
                    "coordinating_residues must be a dict, got "
                    f"{type(coordinating_residues).__name__}"
                )
            if not all(isinstance(k, int) for k in coordinating_residues):
                raise TypeError(
                    f"coordinating_residues keys must be ints (atom IDs), "
                    f"got {set(
                        type(k).__name__ for k in coordinating_residues
                    )}"
                )
            if not all(
                isinstance(
                    v, mda.AtomGroup
                ) for v in coordinating_residues.values()
            ):
                raise TypeError(
                    "coordinating_residues values must be "
                    "MDAnalysis AtomGroups"
                )
        exclude = self.exclude_resids or exclude_residues
        if isinstance(exclude, int):
            exclude = [exclude]

        if not exclude:
            exclude = []

        restraints = {}
        for metal_id, ligating_atoms in ligand_residues.items():
            if metal_id in metal_atom_ids:
                vertices = []
                for ligating_atom in ligating_atoms:
                    if ligating_atom.resid in exclude:
                        continue
                    if self.ligand and ligating_atom.resname.upper() == (
                        self.ligand.residue_name
                    ):
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

    def _set_protein(self):
        self.protein = self.universe.select_atoms("protein")
        if not self.protein:
            message = (
                "Could not set protein for system using:\n"
                "'self.universe.select_atoms('protein')"
            )
            log.error(message)
            raise RuntimeError(message)

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
                    "This may lead to incorrect identification "
                    "of metal atoms.\n"
                    "Consider fixing your PDB file with e.g. pdb4amber.\n"
                )
                guessed_elements = guess_types(self.universe.atoms.names)
                self.universe.add_TopologyAttr("elements", guessed_elements)
                self.metals = self.universe.select_atoms(
                    f"element {metal.upper()}"
                )
            else:
                raise e

        if len(self.metals) == 0:
            self.metals = self.universe.select_atoms(
                f"element {metal.upper()}"
            )
            if len(self.metals) == 0:
                message = f"No atoms found for metal: {self.recipe.metal}"
                log.error(message)
                raise ValueError(message)

        self.metal_resids = self.metals.resids
        self.metal_atomids = self.metals.atoms.ids
        self.metal_resname = metal
        self.metal_element = metal.capitalize()

    def _get_metal_coordinating_residues(self) -> dict[int, mda.AtomGroup]:
        """Get residues coordinating to metal

        Returns:
            dict[int, mda.AtomGroup]: key: metal atom id,
                                      value: atom group of coordinating
                                             residues
        """
        cutoff = self.recipe.coordination_cut_off
        metal_ligands = {}
        try:
            for i in range(len(self.metal_resids)):
                selection = (
                    "element O or element N or element S"
                    f" and sphzone {cutoff} (resid {self.metal_resids[i]})"
                )
                ligands = self.universe.select_atoms(selection)
                key = self.metal_atomids[i]
                metal_ligands[key] = ligands
        except AttributeError as e:
            if "elements" in str(e):
                log.warning(
                    "\nNo element information found in PDB file.\n"
                    "Guessing element information from atom names.\n"
                    "This may lead to incorrect identification of "
                    "metal coordination.\n"
                    "Consider fixing your PDB file with e.g. pdb4amber.\n"
                )
                guessed_elements = guess_types(self.universe.atoms.names)
                self.universe.add_TopologyAttr("elements", guessed_elements)
                selection = (
                    "element O or element N or element S"
                    f" and sphzone {cutoff} (resid {self.metal_resids[i]})"
                )
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
                conect_lines[f"CONECT{counter}"] = [
                    int(parts[1]), int(parts[2])
                ]
                counter += 1

        seen_bridges = set()

        for bridge in self.disulfide_bridges:
            if not {"resid1", "resid2"} <= bridge.keys():
                message = f"Invalid disulfide bridge entry: {bridge}"
                log.error(message)
                raise ValueError(message)

            r1, r2 = bridge["resid1"], bridge["resid2"]

            if r1 == r2:
                message = (
                    f"Disulfide bridge cannot connect residue {r1} to itself."
                )
                log.error(message)
                raise ValueError(message)

            pair = tuple(sorted((r1, r2)))

            if pair in seen_bridges:
                message = f"Duplicate disulfide bridge: {pair}"
                log.error(message)
                raise ValueError(message)
            seen_bridges.add(pair)

            try:
                cyx1 = self.universe.select_atoms(f"resid {r1}").residues[0]
                cyx2 = self.universe.select_atoms(f"resid {r2}").residues[0]

            except IndexError:
                message = (
                    f"Residue {r1} or {r2} not found in structure."
                )
                log.error(message)
                raise ValueError(message)

            if cyx1.resname != "CYX" or cyx2.resname != "CYX":
                message = (
                    f"Disulfide bonds require CYX residues. "
                    f"Got {cyx1.resname} and {cyx2.resname} for {r1} and {r2}."
                )
                log.error(message)
                raise ValueError(message)

            sg1 = cyx1.atoms.select_atoms("name SG")
            sg2 = cyx2.atoms.select_atoms("name SG")

            if len(sg1) == 0 or len(sg2) == 0:
                message = (f"Missing SG atom in residues {r1} or {r2}.")
                log.error(message)
                raise ValueError(message)

            _, _, dists = MDAnalysis.analysis.distances.dist(sg1, sg2)
            distance = dists[0]
            if distance > 3.0:
                message = (
                    f"Disulfide {r1}-{r2} too long: "
                    f"{distance:.2f} Å (likely incorrect)."
                )
                log.error(message)
                raise ValueError(message)

            for _, ids in conect_lines.items():
                if sg1.ids[0] in ids or sg2.ids[0] in ids:
                    log.warning(
                        f"Residues {r1} and {r2} appear to "
                        "already have a disulfide bond "
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
            system=input_system,
            protocol=protocol,
            work_dir=run_directory,
            name=process_name,
            extra_options=config_options,
            extra_lines=namelist_options,
            is_gpu=is_gpu,
            exe=recipe.path_to_engine
        )
        new_topology = os.path.join(
            run_directory,
            f"{process_name}.prm7"
        )
        new_coordinates = os.path.join(
            run_directory,
            f"{process_name}.rst7"
        )

        new_meze = dataclasses.replace(
            self,
            topology=new_topology,
            coordinates=new_coordinates
        )

        if self.restraint_file and os.path.isfile(self.restraint_file):
            new_restraints_file = os.path.join(
                run_directory,
                os.path.basename(self.restraint_file)
            )
            remapped = self._remap_restraint_indices(
                new_meze, outfile=new_restraints_file
            )
            self.restraint_file = remapped
            with open(self.restraint_file, "r") as file:
                added_distance_restraints = file.readlines()
            distance_restraints = (
                distance_restraints or []
            ) + added_distance_restraints

        self.universe = mda.Universe(
            new_topology,
            new_coordinates,
            topology_format="PARM7",
            format="RESTRT"
        )

        if self.recipe.model == 0:
            coordination_restraints = self._prepare_distance_restraints()
            angle_restraints = self._prepare_angle_restraints()
            distance_restraints = (
                distance_restraints or []
            ) + coordination_restraints + angle_restraints

        if distance_restraints:
            self.write_restrained_atoms_pdb(
                output_path=os.path.join(
                    run_directory, f"{process_name}_restrained_atoms.pdb"
                ),
                restraints=distance_restraints
            )
            config_file = process._config_file
            restraint_file = os.path.join(recipe.workdir, "restraints.RST")

            with open(restraint_file, "w") as file:
                file.writelines(distance_restraints)

            step_restraint_file = os.path.join(run_directory, "restraints.RST")
            shutil.copyfile(restraint_file, step_restraint_file)

            with open(config_file, "a") as file:
                file.write(
                    "&wt TYPE='DUMPFREQ', "
                    f"istep1={distance_write_frequency} /\n"
                )

            if not namelist_options:
                with open(config_file, "a") as file:
                    file.write("&wt TYPE=\"END\", /\n")

            with open(config_file, "a") as file:
                file.write("\n")
                file.write("DISANG=restraints.RST\n")
                file.write("DUMPAVE=distances.out\n")

        process.start()
        process.wait()

        if process.isError():
            error_message = (
                f"The run {process_name} exited with an error."
                f"\n\nCheck the log/error files at:\n\t\t{run_directory}\n"
            )
            log.error(error_message)
            info_message = "Outputting the last error and output lines:\n"
            log.info(info_message)
            process.stdout(n=20)
            process.stderr(n=20)

            error_line = process.getStderr()[-1]

            raise RuntimeError(f"{error_message}\n{error_line}")

        new_system = process.getSystem()
        topology, new_coordinates = bss.IO.saveMolecules(
            f"{run_directory}/next",
            system=new_system,
            fileformat=["prm7", "rst7"]
        )

        return dataclasses.replace(
            self,
            topology=topology,
            coordinates=new_coordinates,
            recipe=recipe
        )

    def build_custom_distance_restraints(
        self,
        atom_pairs: list[tuple[str, str]],
        equilibrium_distances: Optional[Union[float, list[float]]] = None,
        force_constant: Optional[Union[float, list[float]]] = 100.0,
        flat_bottom_radius: Optional[Union[float, list[float]]] = 1.0,
    ) -> dict[tuple[int, int], tuple[float, float, float]]:

        n_atom_pairs = len(atom_pairs)

        def _expand(val, name):
            if isinstance(val, float):
                return [val] * n_atom_pairs
            if len(val) != n_atom_pairs:
                message = (
                    f"{name} has {len(val)} values but atom_pairs has"
                    f" {n_atom_pairs} pairs."
                )
                log.error(message)
                raise ValueError(message)
            return val

        force_constant = _expand(force_constant, "force_constant")
        flat_bottom_radii = _expand(flat_bottom_radius, "flat_bottom_radius")

        if equilibrium_distances is not None:
            eq_distances = _expand(
                equilibrium_distances, "equilibrium_distances"
            )

        restraints = {}
        for i, (sel1, sel2) in enumerate(atom_pairs):
            ag1 = self.universe.select_atoms(sel1)
            ag2 = self.universe.select_atoms(sel2)
            if len(ag1) != 1 or len(ag2) != 1:
                message = (
                    f"Each selection must match exactly one atom. "
                    f"'{sel1}' matched {len(ag1)}, "
                    f"'{sel2}' matched {len(ag2)}."
                )
                log.error(message)
                raise ValueError(message)
            dist = eq_distances[i] if equilibrium_distances is not None else \
                MDAnalysis.analysis.distances.dist(ag1, ag2)[-1][0]

            restraints[(ag1.ids[0], ag2.ids[0])] = (
                round(dist, 2),
                round(force_constant[i], 2),
                round(flat_bottom_radii[i], 2)
            )
        return restraints

    def get_ligand_resid(self):
        return self.universe.select_atoms(
            f"resname {self.ligand.residue_name}"
        ).resids[0]

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
                    f"Non-standard residue '{residue}' must have "
                    "'charge' and 'atom_type' properties."
                )
            if not isinstance(properties["charge"], int):
                raise ValueError(
                    f"Non-standard residue '{residue}' has invalid 'charge': "
                    f"{properties['charge']}"
                )
            if properties["atom_type"] not in ["amber", "gaff", "gaff2"]:
                log.warning(
                     f"Non-standard residue '{residue}' has potentially "
                     "unsupported 'atom_type': {properties['atom_type']}"
                )

    def parameterise_non_standard_residues(
            self,
            directory: str,
            non_standard_parameterisation_method: Literal[
                "antechamber", "tleap"
            ] = "antechamber"
    ) -> Optional[list[Ligand]]:
        if non_standard_parameterisation_method not in (
            "antechamber", "tleap"
        ):
            raise ValueError(
                f"non_standard_parameterisation_method must be one of {(
                    'antechamber', 'tleap'
                )} "
                f"got {type(non_standard_parameterisation_method).__name__}"
            )
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
                            "Could not convert residue id "
                            f"{residue} to integer:"
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
            directory: Optional[str] = None,
            mcpbpy_tleap_file: str | None = None,
            non_standard_parameterisation_method: Literal[
                "antechamber", "tleap"
            ] = "antechamber"
    ) -> Self:
        _check_ambertools()
        if non_standard_parameterisation_method not in (
            "antechamber", "tleap"
        ):
            raise ValueError(
                f"non_standard_parameterisation_method must be one of {(
                    'antechamber', 'tleap'
                )} "
                f"got {type(non_standard_parameterisation_method).__name__}"
            )
        if directory:
            os.makedirs(directory, exist_ok=True)

        self._validate_disulfide_bridges()

        if self.recipe.model == 0:
            if self.ligand:
                parameterised_ligand = self.ligand.parameterise(directory)
            else:
                parameterised_ligand = None
            if self.non_standard_residues:
                parameterised_non_standard_residues = (
                    self.parameterise_non_standard_residues(
                        directory=directory,
                        non_standard_parameterisation_method=(
                            non_standard_parameterisation_method
                        )
                    )
                )
        elif self.recipe.model == 2:
            parameterised_ligand = self.ligand
            parameterised_non_standard_residues = self.non_standard_residues
        else:
            raise NotImplementedError(
                f"Model option {self.recipe.model} is not implemented"
            )

        if not mcpbpy_tleap_file:
            tleap_input_file = os.path.join(directory, "tleap_solvate.in")
            tleap_output_file = os.path.join(directory, "tleap_solvate.out")

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

            solvated_complex_topology = (
                f"{parameterised_ligand.name}_complex_solv.prmtop"
            )
            solvated_complex_coordinates = (
                f"{parameterised_ligand.name}_complex_solv.inpcrd"
            )

        else:
            tleap_input_file = mcpbpy_tleap_file
            tleap_output_file = os.path.join(directory, "tleap_solvate.out")
            # TODO disulfide bridges?
            tleap_lines = _edit_mcpbpy_tleap_input(tleap_input_file)

            saveline = [
                line for line in tleap_lines
                if "saveamberparm" in line and "solv" in line
            ][0]
            components = saveline.split()
            solvated_complex_topology = components[2]
            solvated_complex_coordinates = components[3]

            with open(tleap_input_file, "w") as ifile:
                ifile.writelines(tleap_lines)

        workdir = os.getcwd()
        os.chdir(directory)
        tleap_command = f"tleap -s -f {tleap_input_file} > {tleap_output_file}"
        log.info("Running tleap with command:")
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
                non_standard_residues=parameterised_non_standard_residues,
                tleap_input_file=tleap_input_file,
                parameterisation_directory=directory
            )

        except FileNotFoundError:
            log.error("Failed to solvate meze.")
            raise RuntimeError("Failed to solvate meze.")

        os.chdir(workdir)

        if self.restraint_file and os.path.isfile(self.restraint_file):
            remapped = self._remap_restraint_indices(solvated_meze)
            solvated_meze = dataclasses.replace(
                solvated_meze, restraint_file=remapped
            )

        return solvated_meze

    def _remap_restraint_indices(
            self,
            updated_meze: "Meze",
            outfile: Optional[str] = None
    ):

        with open(self.restraint_file, "r") as file:
            lines = file.readlines()

        restraint_atom_ids: list[int] = []
        for line in lines:
            if "iat=" in line:
                after_iat = line.split("iat=")[1]
                raw_indices = after_iat.split(",")[:2]
                restraint_atom_ids.extend(int(iat) for iat in raw_indices)

        atom_index_map: dict[int, int] = {}
        for old_id in restraint_atom_ids:
            ag = self.universe.select_atoms(f"id {old_id}")
            if len(ag) == 0:
                log.warning(
                    f"Atom id {old_id} from restraint file "
                    "not found in pre-solvation universe."
                )
                continue
            resid = ag.atoms[0].resid
            resname = ag.atoms[0].resname
            name = ag.atoms[0].name
            new_ag = updated_meze.universe.select_atoms(
                f"resid {resid} and name {name}"
            )
            if len(new_ag) == 0:
                log.warning(
                    f"Could not find atom by resid={resid} and name={name}"
                    " in updated meze object.\n"
                    f"Falling back to resname={resname} and name={name}."
                )
                new_ag = updated_meze.universe.select_atoms(
                    f"resname {resname} and name {name}"
                )
                if len(new_ag) == 0:
                    log.warning(
                        f"Could not find atom (resname={resname}, name={name})"
                        " in updated meze object."
                        f"Restraint at index {old_id} will not be remapped."
                    )
                    continue
                if len(new_ag) > 1:
                    log.warning(
                        f"Multiple atoms found with resname={resname}, "
                        f"name={name} in updated meze object."
                        f"Restraint at index {old_id} will not be remapped."
                    )
                    continue
            if old_id == new_ag.atoms[0].id:
                log.info(
                    f"Index {old_id} from restraint file matches"
                    " current meze object.\n"
                    "Will not perform remapping."
                )
                continue
            atom_index_map[old_id] = new_ag.atoms[0].id

        new_lines = []
        for line in lines:
            if "iat=" in line:
                after_iat = line.split("iat=")[1]
                raw_indices = after_iat.split(",")[:2]
                old_a, old_b = int(raw_indices[0]), int(raw_indices[1])
                new_a = atom_index_map.get(old_a, old_a)
                new_b = atom_index_map.get(old_b, old_b)
                line = line.replace(
                    f"iat={old_a},{old_b},", f"iat={new_a},{new_b},"
                )
                if new_a != old_a or new_b != old_b:
                    log.info(
                        f"Replacing 'iat={old_a},{old_b}'"
                        f" with 'iat={new_a},{new_b}'"
                    )
            new_lines.append(line)

        if not outfile:
            restraint_file = updated_meze.restraint_file
        else:
            restraint_file = outfile

        with open(restraint_file, "w") as file:
            file.writelines(new_lines)

        return restraint_file

    def prepare_mcpb_system(
            self,
            directory: str | None = None,
            ligand_name: str = "ligand"
    ) -> Self:

        if directory:
            os.makedirs(directory, exist_ok=True)

        parameterisation_directory = os.path.join(
            directory, "01_mcpb_parameterisation"
        )
        os.makedirs(parameterisation_directory, exist_ok=True)

        self._validate_disulfide_bridges()

        parameterised_ligand = self.ligand.parameterise(
            directory=parameterisation_directory,
            filename="MOL"
        )

        self.prepare_metals_for_ezaff(directory=parameterisation_directory)

        parameterised_non_standard_residues = (
            self.parameterise_non_standard_residues(
                directory=parameterisation_directory
            )
        )

        complex = self.write_complex(
            directory=parameterisation_directory,
            ligand_name=ligand_name,
            parameterised_non_standard_residues=(
                parameterised_non_standard_residues
            )
        )

        return dataclasses.replace(
            self,
            parameterisation_directory=parameterisation_directory,
            topology=complex.filename,
            coordinates=complex.filename,
            ligand=parameterised_ligand,
            non_standard_residues=parameterised_non_standard_residues,
        )

    def write_mcpb_input_file(
            self,
            directory: str,
            original_pdb: str,
            parameterised_ligand: Ligand,
            parameterised_non_standard_residues: Optional[list[Ligand]],
            metals: list[str],
            ligand_name: str = "ligand"
    ) -> str:
        if parameterised_non_standard_residues is not None:
            non_ligands = [
                residue for residue in parameterised_non_standard_residues
                if not isinstance(residue, Ligand)
            ]
            if non_ligands:
                raise TypeError(
                    "parameterised_non_standard_residues must be"
                    " a list of Ligand objects, "
                    f"got {
                        [type(residue).__name__ for residue in non_ligands]
                    }"
                )
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
            "naa_mol2files": (
                f"{parameterised_ligand.name}.mol2"
                if not parameterised_non_standard_residues else
                " ".join(
                    [parameterised_ligand.name + ".mol2"] +
                    [residue.name + ".mol2"
                     for residue in parameterised_non_standard_residues]
                )
            ),
            "frcmod_files": (
                f"{parameterised_ligand.name}.frcmod"
                if not parameterised_non_standard_residues else " ".join(
                    [parameterised_ligand.name + ".frcmod"] +
                    [residue.name + ".frcmod"
                     for residue in parameterised_non_standard_residues]
                )
            ),
            "software_version": self.recipe.gaussian_version,
            "ion_ids": " ".join(str(atomid) for atomid in self.metal_atomids),
            "large_opt": int(self.recipe.only_optimise_hydrogens),
            "force_field": self.recipe.protein_forcefield,
            "water_model": self.recipe.water_model,
            "gaff": gaff
        }

        with open(mcpb_input_file, "w") as mcpb_file:
            for key, value in mcpb_input_options.items():
                mcpb_file.write(f"{key} {value}\n")

        return mcpb_input_file

    def prepare_resp_calculation(
            self,
            ligand_name: str = "ligand",
            split_large_files: bool = True,
            sbatch_options: Optional[dict] = None,
            additional_lines: Optional[list[str]] = None
    ):
        _check_ambertools()
        if additional_lines is not None and not all(
            isinstance(line, str) for line in additional_lines
        ):
            raise TypeError("additional_lines must be a list of strings")

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
            parameterised_non_standard_residues=(
                self.non_standard_residues
                if isinstance(self.non_standard_residues, list) else None
            ),
            metals=metals,
            ligand_name=ligand_name
        )

        workdir = os.getcwd()
        os.chdir(self.parameterisation_directory)

        mcpb_output_file = os.path.join(
            self.parameterisation_directory, "mcpb_step1.out"
        )
        mcpb_command = (
            f"MCPB.py -i {mcpb_input_file} -s 1 > {mcpb_output_file}"
        )
        log.info(f"Running MCPB.py step 1 with command:\n{mcpb_command}")
        os.system(mcpb_command)

        com_files = sorted(
            glob.glob(f"{self.parameterisation_directory}/*.com")
        )

        if not com_files:
            raise ValueError(
                "No Gaussian .com files found in "
                f"{self.parameterisation_directory}."
            )

        if split_large_files:
            self.update_gaussian_inputs(
                directory=self.parameterisation_directory
            )
            com_files = sorted(
                glob.glob(f"{self.parameterisation_directory}/*.com")
            )
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

                    calculation_line = [
                        line for line in large_lines if "#" in line
                    ][0]
                    replace_index = large_lines.index(calculation_line)

                    split_parts = calculation_line.split("Opt", 1)
                    level_of_theory = split_parts[0]
                    pop_analysis = split_parts[1]
                    large_optimisation_line = level_of_theory + "Opt\n"
                    population_analysis_line = (
                        level_of_theory +
                        "guess=read geom=checkpoint" +
                        pop_analysis
                    )

                    clear_line = [
                        i for i, line in enumerate(large_lines)
                        if "CLR" in line
                    ][0]
                    header_end = clear_line + 3

                    large_opt_lines = large_lines.copy()
                    large_opt_lines[replace_index] = large_optimisation_line

                    large_mk_lines = large_lines[:header_end]
                    large_mk_lines[replace_index] = population_analysis_line

                    large_opt_file = large_file.replace(
                        "large_mk", "large_opt"
                    )

                    large_opt_lines = [
                        line.replace(line, "")
                        if self.metal_element in line and len(
                            line.split()
                        ) < 3 else line for line in large_opt_lines
                    ]

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
                    new_lines.append(
                        f"%NProcShared={self.recipe.nprocshared}\n"
                    )
                else:
                    new_lines.append(line)

            with open(com_file, "w") as file:
                file.writelines(new_lines)

    def prepare_metals_for_ezaff(self, directory: str) -> List[str]:
        _check_ambertools()
        metals = []
        for i, metal in enumerate(self.metals):
            metal_atomgroup = self.universe.select_atoms(
                f"resid {metal.resid}"
            )
            metal_mcpb_resname = f"{metal.name.upper()}{i+1}"

            metal_atomgroup.write(f"{directory}/{metal_mcpb_resname}.pdb")

            metal_to_pdb_command = (
                f"metalpdb2mol2.py -i {directory}/{metal_mcpb_resname}.pdb "
                f"-o {directory}/{metal_mcpb_resname}.mol2 "
                f"-c {self.recipe.metal_charge}"
            )

            os.system(metal_to_pdb_command)
            metals.append(f"{metal_mcpb_resname}.mol2")

        return metals

    def write_complex(
            self,
            directory: str,
            ligand_name: str = "ligand",
            parameterised_non_standard_residues: List[Ligand] = None
    ) -> dict[str, str]:
        _check_ambertools()

        ligand_file = os.path.join(directory, f"{self.ligand.name}.pdb")
        ligand_universe = mda.Universe(ligand_file)
        ligand = ligand_universe.atoms

        non_standard_residues = list(
            self.non_standard_residues.keys()
        ) if self.non_standard_residues else []
        if 0 < len(non_standard_residues) <= 1:
            non_standard_residues = self.universe.select_atoms(
                f"resname {non_standard_residues[0]}"
            )
        elif len(non_standard_residues) > 1:
            non_standard_residues = self.universe.select_atoms(
                " or ".join(f"resname {res}" for res in non_standard_residues)
            )
        else:
            non_standard_residues = None

        components = [self.protein, self.metals, ligand]
        if non_standard_residues is not None:
            components.append(non_standard_residues)
        components.append(self.crystal_waters)

        merged = mda.Merge(*components)
        merged.atoms.write(f"{directory}/{ligand_name}_complex.pdb")

        pdb4amber_command = (
            f"pdb4amber -i {directory}/{ligand_name}_complex.pdb "
            f"-o {directory}/{self.recipe.group_name}_{ligand_name}.amber.pdb"
        )

        log.info(f"Running pdb4amber with command:\n{pdb4amber_command}")
        os.system(pdb4amber_command)
        merged.filename = os.path.join(
            directory,
            f"{self.recipe.group_name}_{ligand_name}.amber.pdb"
        )
        return merged

    def build_empirical_bonds(self):
        _check_ambertools()
        mcpbpy_input_file = self.mcpbpy_input_file

        workdir = os.getcwd()
        os.chdir(self.parameterisation_directory)
        step_2e_output_file = os.path.join(
            self.parameterisation_directory, "mcpb_step2e.out"
        )
        step_2e_command = (
            f"MCPB.py -i {mcpbpy_input_file} "
            f"-s 2e > {step_2e_output_file}"
        )
        log.info(f"Running MCPB.py step 2e with command:\n{step_2e_command}")
        os.system(step_2e_command)
        os.chdir(workdir)
        return dataclasses.replace(
            self,
            restraint_file=self.restraint_file
        )

    def _remove_double_oxygen_bond(
            self, parameterisation_directory: str
    ) -> str | None:
        _check_ambertools()

        tleap_file = glob.glob(
            f"{parameterisation_directory}/*tleap.in"
        )
        if len(tleap_file) == 0:
            raise FileNotFoundError(
                "Cannot find tleap input file: "
                f"{parameterisation_directory}/*tleap.in"
            )

        tleap_file = tleap_file[0]

        if not os.path.isfile(tleap_file + "_unedited"):
            shutil.copy(
                tleap_file,
                tleap_file + "_unedited"
            )

        with open(tleap_file, "r") as ifile:
            all_lines = ifile.readlines()

        oxygen_ligands = {}
        for metal, ligands in self.coordinating_residues.items():
            metal_resid = self.universe.select_atoms(f"id {metal}")[0].resid
            oxygens = [atom for atom in ligands if atom.element == "O"]
            if len(oxygens) > 1:
                oxygen_ligands[metal_resid] = oxygens

        if not oxygen_ligands:
            log.info("No metals with multiple oxygen ligands found.")
            return self.restraint_file

        oxygen_resids = {
            atom.resid
            for oxygens in oxygen_ligands.values() for atom in oxygens
        }
        metal_resids = set(oxygen_ligands.keys())

        def parse_bond(line):
            parts = line.replace("bond", "").strip().split("mol.")
            r1, a1 = parts[1].strip().split(".")[:2]
            r2, a2 = parts[2].strip().split(".")[:2]
            return int(r1), a1.strip(), int(r2), a2.strip()

        metal_bond_lines = [
            line for line in all_lines if "bond" in line
            and self.metal_element.upper() in line
        ]
        bonds_to_remove = []
        for line in metal_bond_lines:
            r1, a1, r2, a2 = parse_bond(line)
            if (r1 in oxygen_resids or r2 in oxygen_resids) and \
               (r1 in metal_resids or r2 in metal_resids):
                bonds_to_remove.append((line, r1, a1, r2, a2))
                log.info(f"Found metal-oxygen bond to remove:\n{line}")

        if not bonds_to_remove:
            log.info("No double oxygen bonds found to remove.")
            return self.restraint_file

        lines_to_remove = {bond[0] for bond in bonds_to_remove}
        with open(tleap_file, "w") as file:
            file.writelines(
                line for line in all_lines if line not in lines_to_remove
            )

        standard_fingerprint_file = glob.glob(
            f"{parameterisation_directory}/*standard.fingerprint"
        )
        if len(standard_fingerprint_file) == 0:
            raise FileNotFoundError(
                "Cannot find standard fingerprint file: "
                f"{parameterisation_directory}/*standard.fingerprint"
            )

        standard_fingerprint_file = standard_fingerprint_file[0]

        with open(standard_fingerprint_file) as ifile:
            fingerprint_lines = ifile.readlines()

        def get_mcpb_atom_type(resid, atom_name):
            for line in fingerprint_lines:
                if "->" not in line:
                    continue
                parts = line.split()[0].split("-")
                if parts[0] == str(resid) and parts[2] == atom_name:
                    return line.strip().split("->")[-1].strip()
            return None

        frcmod_file = glob.glob(
            f"{parameterisation_directory}/*mcpbpy.frcmod"
        )
        if len(frcmod_file) == 0:
            raise FileNotFoundError(
                "Cannot find MCPB.py frcmod file: "
                f"{parameterisation_directory}/*mcpbpy.frcmod"
            )

        frcmod_file = frcmod_file[0]

        with open(frcmod_file, "r") as ifile:
            frcmod_lines = ifile.readlines()

        bond_start = [
            i for i, line in enumerate(frcmod_lines) if "BOND" in line
        ][0] + 1
        angle_start = [
            i for i, line in enumerate(frcmod_lines) if "ANGL" in line
        ][0]

        frcmod_bond_lines = frcmod_lines[bond_start:angle_start]

        def get_bond_params(type1, type2):
            for line in frcmod_bond_lines:
                if not line.strip():
                    continue
                parts = line.split()
                a, b = parts[0].split("-")
                if {a, b} == {type1, type2}:
                    return float(parts[1]), float(parts[2])
            return None, None

        restraints = {}
        for _, r1, a1, r2, a2 in bonds_to_remove:
            type1 = get_mcpb_atom_type(r1, a1)
            type2 = get_mcpb_atom_type(r2, a2)
            fc, eq = get_bond_params(type1, type2)
            id1 = self.universe.select_atoms(f"resid {r1} and name {a1}")[0].id
            id2 = self.universe.select_atoms(f"resid {r2} and name {a2}")[0].id
            metal_id, o_id = (id1, id2) if r1 in metal_resids else (id2, id1)
            restraints[(metal_id, o_id)] = (
                np.round(eq, 2), np.round(fc, 2), 1.0
            )

        restraint_lines = _write_distance_restraints(restraints)
        restraint_file = os.path.join(
            parameterisation_directory, "double_oxygen_restraints.RST"
        )
        if os.path.isfile(restraint_file):
            with open(restraint_file, "r") as file:
                read_lines = file.readlines()
            if not read_lines:
                with open(restraint_file, "w") as file:
                    for lines in restraint_lines:
                        file.writelines(lines)
        else:
            with open(restraint_file, "w") as file:
                for lines in restraint_lines:
                    file.writelines(lines)
        log.info(
            f"Added harmonic restraints for deleted bonds"
            f" between metal and oxygen ligand(s) to {restraint_file}."
        )
        return restraint_file

    def _remove_ligand_bond(self, parameterisation_directory: str) -> None:

        tleap_file = glob.glob(
            f"{parameterisation_directory}/*tleap.in"
        )
        if len(tleap_file) == 0:
            raise FileNotFoundError(
                "Cannot find tleap input file: "
                f"{parameterisation_directory}/*tleap.in"
            )

        tleap_file = tleap_file[0]
        if not os.path.isfile(tleap_file + "_unedited"):
            shutil.copy(
                tleap_file,
                tleap_file + "_unedited"
            )

        with open(tleap_file, "r") as ifile:
            all_lines = ifile.readlines()

        bond_lines = [
            line for line in all_lines
            if "bond" in line and f"{self.metal_element.upper()}" in line
        ]

        ligand_metal_coordination_lines = []
        for line in bond_lines:
            parts = line.replace("bond", "").strip().split("mol.")
            first_atom = parts[1].strip()
            second_atom = parts[2].strip()
            first_resid = int(first_atom.split(".")[0])
            second_resid = int(second_atom.split(".")[0])

            if (
                self.ligand_resid == first_resid or
                self.ligand_resid == second_resid
            ):
                ligand_metal_coordination_lines.append(line)
                log.info(f"Found metal-ligand bond:\n{line}")

        if not ligand_metal_coordination_lines:
            log.info(
                "Did not find a bond between the ligand"
                f"{self.ligand.residue_name} and the metal"
            )
        else:
            new_lines = [
                line for line in all_lines
                if line not in ligand_metal_coordination_lines
            ]

            with open(tleap_file, "w") as ofile:
                ofile.writelines(new_lines)

            for line in ligand_metal_coordination_lines:
                log.info(
                    "Succesfully removed bond: "
                    f"{line}"
                )

    def build_resp_charges(
            self,
            fix_ligand_charge: bool = True,
            directory: Optional[str] = None
    ):
        _check_ambertools()
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
                    "parent directory not set, "
                    f"inferring from {self.parameterisation_directory}"
                )
                directory = str(
                    pathlib.Path(self.parameterisation_directory).parent
                )

            parameterisation_directory = os.path.join(
                directory, "02_fixed_ligand_charge"
            )
            parameterisation_directory = parameterisation_directory
            log.info(f"Creating directory: {parameterisation_directory}")
            os.makedirs(parameterisation_directory, exist_ok=True)

            ligand_files = glob.glob(
                os.path.join(
                    self.parameterisation_directory,
                    f"{self.ligand.residue_name}.*"
                )
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

            non_standard_residue_files = [
                res.file[0] for res in self.non_standard_residues
            ]
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

            checkpoint_files = glob.glob(
                f"{self.parameterisation_directory}/*.chk"
            )

            param_files = (
                ligand_files + non_standard_residue_files + original_zn_files +
                log_files + frcmod_files + checkpoint_files + large_pdb_file +
                large_fingerprint + standard_pdb +
                [
                    original_pdb_file,
                    standard_fingerprint,
                    mcpbpy_input_file,
                    self.restraint_file
                ]
            )

            new_zn_files = []
            new_restraints = self.restraint_file
            for old_file in param_files:
                file = os.path.basename(old_file)
                new_file = os.path.join(parameterisation_directory, file)
                shutil.copy(old_file, new_file)
                if "ZN" in file:
                    new_zn_files.append(new_file)
                if ".RST" in os.path.splitext(file):
                    new_restraints = new_file

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
            new_restraints = self.restraint_file

        step_3_command = (
            f"MCPB.py -i {mcpbpy_input_file} -s 3 > {step_3_output_file}"
        )
        log.info(f"Running MCPB.py step 3 with command:\n{step_3_command}")
        workdir = os.getcwd()
        os.chdir(parameterisation_directory)
        os.system(step_3_command)

        step_4_command = (
            f"MCPB.py -i {mcpbpy_input_file} -s 4 > {step_4_output_file}"
        )
        log.info(f"Running MCPB.py step 4 with command:\n{step_4_command}")
        os.system(step_4_command)
        os.chdir(workdir)

        tleap_file = glob.glob(f"{parameterisation_directory}/*tleap.in")[0]
        if not tleap_file:
            raise RuntimeError(
                "No tleap input file found after MCPB.py step 4. "
                f"Check log file: {step_4_output_file}"
            )

        self._remove_ligand_bond(parameterisation_directory)
        self.restraint_file = self._remove_double_oxygen_bond(
            parameterisation_directory
        )

        new_coordinates = glob.glob(
            f"{parameterisation_directory}/*_mcpbpy.pdb"
        )
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
            os.path.join(
                parameterisation_directory,
                f"{
                    self.ligand.residue_name[0] +
                    self.ligand.residue_name[-1]
                }.mol2"
            )
        )[0]

        new_ligand_frcmod_file = glob.glob(
            f"{parameterisation_directory}/{self.ligand.residue_name}.frcmod"
        )[0]

        new_ligand_resname = pathlib.Path(new_ligand_file).stem
        new_ligand = Ligand(
            new_ligand_file,
            charge=_get_mol2_charge(new_ligand_file),
            parameterised=True,
            residue_name=new_ligand_resname,
            frcmod_file=new_ligand_frcmod_file
        )

        new_non_standard_files = [glob.glob(
            os.path.join(
                parameterisation_directory,
                f"{
                    residue.residue_name[0] +
                    residue.residue_name[-1]
                }*.mol2"
            )
        )[0] for residue in self.non_standard_residues]
        non_standard_frcmod_files = [glob.glob(
            f"{parameterisation_directory}/{residue.residue_name}.frcmod"
        )[0] for residue in self.non_standard_residues]

        new_non_standard_resnames = [
            pathlib.Path(file).stem for file in new_non_standard_files
        ]
        new_non_standard_charges = [
            _get_mol2_charge(file) for file in new_non_standard_files
        ]
        new_non_standard_residues = [Ligand(
            file=mol2,
            charge=charge,
            parameterised=True,
            residue_name=name,
            frcmod_file=frcmod
        ) for mol2, charge, name, frcmod in zip(
            new_non_standard_files,
            new_non_standard_charges,
            new_non_standard_resnames,
            non_standard_frcmod_files
        )]

        return dataclasses.replace(
            self,
            tleap_input_file=tleap_file,
            parameterisation_directory=parameterisation_directory,
            mcpbpy_input_file=mcpbpy_input_file,
            coordinates=new_coordinates,
            topology=new_coordinates,
            ligand=new_ligand,
            non_standard_residues=new_non_standard_residues,
            restraint_file=new_restraints
        )


@dataclass
class ColdMeze(Meze):
    recipe: ColdMezeRecipe
    exclude_resids: Optional[Union[int, list[int]]] = field(
        default_factory=list
    )
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
        recipe: Optional[Union[dict, "ColdMezeRecipe"]] = None,
        disulfide_bridges: Optional[List[dict[str, int]]] = None,
        ligand: Optional[Ligand] = None,
        ligand_resid: Optional[int] = None,
        non_standard_residues: dict[dict] | List[Ligand] = None,
        parameterisation_directory: Optional[str] = None,
        mcpbpy_input_file: Optional[str] = None,
        tleap_input_file: Optional[str] = None,
        restraint_file: Optional[str] = None,
        exclude_resids: Optional[Union[int, list[int]]] = None,
        ligand_resname: Optional[str] = None,
        stage: Optional[str] = "bound",
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
            message = (
                "You must supply either a pdb file or "
                "both a topology and coordinate file."
            )
            log.error(message)
            raise ValueError(message)
        if recipe is None:
            recipe = ColdMezeRecipe(**kwargs)
        elif isinstance(recipe, dict):
            recipe = ColdMezeRecipe(**recipe)
        elif not isinstance(recipe, ColdMezeRecipe):
            message = (
                "Expected 'recipe' to be a ColdMezeRecipe, dict, or None,"
                f" but got {type(recipe).__name__}"
            )
            log.error(message)
            raise TypeError(message)

        return cls(
            topology=topology,
            coordinates=coordinates,
            disulfide_bridges=disulfide_bridges,
            ligand=ligand,
            ligand_resid=ligand_resid,
            non_standard_residues=non_standard_residues,
            parameterisation_directory=parameterisation_directory,
            mcpbpy_input_file=mcpbpy_input_file,
            tleap_input_file=tleap_input_file,
            restraint_file=restraint_file,
            exclude_resids=exclude_resids,
            ligand_resname=ligand_resname,
            recipe=recipe,
            stage=stage
        )

    def _build_restraint_mask(
            self,
            position_restraints: str,
            exclude_resids: Optional[Union[int, list[int]]] = [],
            additional_restraints: Optional[dict[str, Any]] = None
    ) -> str | None:
        """Build an amber-compatible restraint mask

        Args:
            position_restraints (str): what type of position
                                       restraints to apply
            additional_restraints (Optional[dict[str, Any]]):
                                Additional restraints to apply

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

        if self.coordinating_residues:
            coordinating_atomgroups = next(
                iter(self.coordinating_residues.values())
            )
            for atomgroup in list(self.coordinating_residues.values())[1:]:
                coordinating_atomgroups += atomgroup

            coordinating_resids = [
                atom.resid for atom in coordinating_atomgroups
                if atom.resid not in exclude_resids
            ]
        else:
            coordinating_resids = []
        additional_resids = []
        if additional_restraints:
            if not (
                {"resids"} <= additional_restraints.keys()
                or {"resnames"} <= additional_restraints.keys()
            ):
                message = (
                    "additional_restraints must contain "
                    "'resids' or 'resnames' keys."
                )
                log.error(message)
                raise ValueError(message)

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
                    atom.resid for atom in self.universe.select_atoms(
                        f"resname {resname}"
                    )
                )
                additional_resids.update(resname_resids)
            additional_resids = list(additional_resids)
        if position_restraints == "solute":
            protein_resids = [
                atom.resid for atom in self.universe.select_atoms("protein")
            ]
            if not protein_resids:
                solute_resids = [
                    atom.resid for atom in self.universe.select_atoms(
                        f"resname {self.ligand_resname}"
                    )
                ]
            else:
                solute_resids = protein_resids
            constraint_resids = (
                solute_resids +
                coordinating_resids +
                list(self.metal_resids) +
                additional_resids
            )
            return f"':{_residue_restraint_mask(constraint_resids)}'"
        elif position_restraints == "backbone":
            constraint_resids = (
                coordinating_resids +
                list(self.metal_resids) +
                additional_resids
            )
            return f"'(@N,CA,C,O & !:WAT)|:{
                _residue_restraint_mask(constraint_resids)
            }'"
        elif position_restraints == "metal-coordination":
            constraint_resids = (
                coordinating_resids +
                list(self.metal_resids) +
                additional_resids
            )
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
            timestep: Optional[Union[float, "bssTime"]] = None,
            runtime: Optional[Union[float, "bssTime"]] = None,
            temperature: Optional[Union[float, "bssTemperature"]] = None,
            start_temperature: Optional[Union[float, "bssTemperature"]] = 300,
            end_temperature: Optional[Union[float, "bssTemperature"]] = 300,
            pressure: Optional[Union[float, "bssPressure"]] = None,
            is_gpu: Optional[bool] = True,
            engine_executable: Optional["str"] = None,
            additional_positional_restraints: Optional[dict[str, Any]] = None,
            additional_distance_restraints: Optional[
                dict[tuple[int, int], tuple[float, float, float]]
            ] = None
    ) -> "ColdMeze":
        allowed_protocols = ["minimisation", "nvt", "npt"]
        if protocol_type not in allowed_protocols:
            raise ValueError(
                f"Unsupported protocol type '{protocol_type}'.\n"
                f"Must be one of {allowed_protocols}."
            )

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
            start_temperature=(
                start_temperature or self.recipe.start_temperature
            ),
            end_temperature=end_temperature or self.recipe.end_temperature,
            pressure=pressure or self.recipe.pressure,
            restraint_weight=restraint_weight or self.recipe.restraint_weight,
            path_to_engine=engine_executable or self.recipe.path_to_engine,
            model=self.recipe.model
        )

        config_options = {
            "cut": recipe.nb_cutoff._value,
            "ntpr": 1000,
            "iwrap": 0
        }

        if restart:
            config_options["irest"] = 1
            config_options["ntx"] = 5

        if position_restraints:
            config_options["restraintmask"] = self._build_restraint_mask(
                position_restraints=position_restraints,
                additional_restraints=additional_positional_restraints
            )

        if (
            self.recipe.model == 0 or self.restraint_file
            or additional_distance_restraints
        ):
            config_options["nmropt"] = 1

        extra_distance_restraints = _write_distance_restraints(
            additional_distance_restraints
        ) if additional_distance_restraints else None

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
                temperature = recipe.temperature

            protocol = bss.Protocol.Equilibration(
                timestep=recipe.dt,
                runtime=recipe.runtime,
                temperature_start=recipe.start_temperature,
                temperature_end=recipe.end_temperature,
                temperature=temperature,
                pressure=None,
                restraint="all" if position_restraints else None,
                force_constant=recipe.restraint_weight
            )
        elif protocol_type == "npt":
            config_options["barostat"] = recipe.barostat
            protocol = bss.Protocol.Equilibration(
                timestep=recipe.dt,
                runtime=recipe.runtime,
                temperature=recipe.temperature,
                pressure=recipe.pressure,
                restraint="all" if position_restraints else None,
                force_constant=recipe.restraint_weight
            )
        return super()._run(
            protocol=protocol,
            recipe=recipe,
            system=system,
            process_name=process_name,
            config_options=config_options,
            is_gpu=is_gpu,
            distance_restraints=extra_distance_restraints
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
            additional_positional_restraints: Optional[dict[str, Any]] = None,
            additional_distance_restraints: Optional[
                dict[tuple[int, int], tuple[float, float, float]]
            ] = None
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
            additional_positional_restraints=additional_positional_restraints,
            additional_distance_restraints=additional_distance_restraints
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
            timestep: Optional[Union[float, "bssTemperature"]] = None,
            runtime: Optional[Union[float, "bssTime"]] = None,
            temperature: Optional[Union[float, "bssTemperature"]] = None,
            start_temperature: Optional[Union[float, "bssTemperature"]] = 300,
            end_temperature: Optional[Union[float, "bssTemperature"]] = 300,
            process_name: Optional[str] = "nvt",
            is_gpu: Optional[bool] = True,
            engine_executable: Optional[str] = None,
            additional_positional_restraints: Optional[dict[str, Any]] = None,
            additional_distance_restraints: Optional[
                dict[tuple[int, int], tuple[float, float, float]]
            ] = None

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
            additional_positional_restraints=additional_positional_restraints,
            additional_distance_restraints=additional_distance_restraints
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
            timestep: Optional[Union[float, "bssTemperature"]] = None,
            runtime: Optional[Union[float, "bssTime"]] = None,
            temperature: Optional[Union[float, "bssTemperature"]] = 300,
            pressure: Optional[Union[float, "bssPressure"]] = 1.0,
            process_name: Optional[str] = "npt",
            is_gpu: Optional[bool] = True,
            engine_executable: Optional[str] = None,
            additional_positional_restraints: Optional[dict[str, Any]] = None,
            additional_distance_restraints: Optional[
                dict[tuple[int, int], tuple[float, float, float]]
            ] = None

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
            additional_positional_restraints=additional_positional_restraints,
            additional_distance_restraints=additional_distance_restraints
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
                message = (
                    f"Restraint file not found: {self.restraint_file}"
                )
                log.error(message)
                raise FileNotFoundError(message)

        elif not self.restraint_file and self.recipe.model == 0:
            log.warning(
                "No restraint file supplied while model is 0.\n"
                "Restraints will be determined from input files."
            )

    @classmethod
    def from_files(
        cls,
        pdb_file: Optional[str] = None,
        topology: Optional[str] = None,
        coordinates: Optional[str] = None,
        recipe: Optional[Union[dict, "HotMezeRecipe"]] = None,
        disulfide_bridges: Optional[List[dict[str, int]]] = None,
        ligand: Optional[Ligand] = None,
        ligand_resid: Optional[int] = None,
        non_standard_residues: dict[dict] | List[Ligand] = None,
        parameterisation_directory: Optional[str] = None,
        mcpbpy_input_file: Optional[str] = None,
        tleap_input_file: Optional[str] = None,
        restraint_file: Optional[str] = None,
        exclude_resids: Optional[Union[int, list[int]]] = None,
        ligand_resname: Optional[str] = None,
        stage: Optional[str] = "bound",
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
                "You must supply either a pdb file or "
                "both a topology and coordinate file."
            )

        if recipe is None:
            recipe = HotMezeRecipe(**kwargs)
        elif isinstance(recipe, dict):
            recipe = HotMezeRecipe(**recipe)
        elif not isinstance(recipe, HotMezeRecipe):
            raise TypeError(
                "Expected 'recipe' to be a HotMezeRecipe, dict, "
                f"or None, but got {type(recipe).__name__}"
            )

        return cls(
            topology=topology,
            coordinates=coordinates,
            disulfide_bridges=disulfide_bridges,
            ligand=ligand,
            ligand_resid=ligand_resid,
            non_standard_residues=non_standard_residues,
            parameterisation_directory=parameterisation_directory,
            mcpbpy_input_file=mcpbpy_input_file,
            tleap_input_file=tleap_input_file,
            restraint_file=restraint_file,
            exclude_resids=exclude_resids,
            ligand_resname=ligand_resname,
            recipe=recipe,
            stage=stage
        )

    def run(
            self,
            workdir: Optional[str],
            system: Optional[bssSystem] = None,
            process_name: Optional[str] = "meze-run",
            nb_cutoff: Optional[float] = None,
            timestep: Optional[Union[float, "bssTime"]] = None,
            runtime: Optional[Union[float, "bssTime"]] = None,
            temperature: Optional[Union[float, "bssTemperature"]] = 300,
            pressure: Optional[Union[float, "bssPressure"]] = 1,
            engine_executable: Optional[str] = None,
            write_frequency: Optional[int] = 100000,
            distance_write_frequency: Optional[int] = 10000,
            additional_distance_restraints: Optional[
                dict[tuple[int, int], tuple[float, float, float]]
            ] = None
    ):
        recipe = HotMezeRecipe(
            workdir=workdir or self.recipe.workdir,
            nb_cutoff=nb_cutoff or self.recipe.nb_cutoff,
            runtime=runtime or self.recipe.runtime,
            dt=timestep or self.recipe.dt,
            temperature=temperature or self.recipe.temperature,
            pressure=pressure or self.recipe.pressure,
            path_to_engine=engine_executable or self.recipe.path_to_engine,
            model=self.recipe.model
        )

        config_options = {"cut": recipe.nb_cutoff._value,
                          "ntpr": write_frequency,
                          "ntwx": write_frequency,
                          "ntwr": write_frequency,
                          "irest": 1,
                          "ntx": 5,
                          "iwrap": 0}

        if (
            self.recipe.model == 0 or self.restraint_file
            or additional_distance_restraints
        ):
            config_options["nmropt"] = 1

        protocol = bss.Protocol.Production(
            timestep=recipe.dt,
            runtime=recipe.runtime,
            temperature=recipe.temperature,
            pressure=recipe.pressure
        )
        if self.restraint_file and os.path.isfile(self.restraint_file):
            step_restraint_file = os.path.join(
                recipe.workdir, "restraints.RST"
            )
            shutil.copyfile(self.restraint_file, step_restraint_file)
        extra_distance_restraints = _write_distance_restraints(
            additional_distance_restraints
        ) if additional_distance_restraints else None
        return super()._run(
            protocol=protocol,
            recipe=recipe,
            system=system,
            process_name=process_name,
            config_options=config_options,
            distance_write_frequency=distance_write_frequency,
            distance_restraints=extra_distance_restraints
        )


@dataclass
class QuantumMeze(Meze):
    recipe: MezeRecipe
    exclude_resids: Optional[Union[int, list[int]]] = field(
        default_factory=list
    )
    metal_resids_for_distance_restraints: Optional[
        Union[int, list[int]]
    ] = None
    additional_qm_resids: Optional[Union[int, list[int]]] = None
    additional_qm_resnames: Optional[Union[str, list[str]]] = None
    custom_qm_region: Optional[dict[str, list]] = None

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.exclude_resids, int):
            self.exclude_resids = [self.exclude_resids]

        if isinstance(self.metal_resids_for_distance_restraints, int):
            self.metal_resids_for_distance_restraints = [
                self.metal_resids_for_distance_restraints
            ]

        self.exclude_resids = set(self.exclude_resids or [])

        if isinstance(self.additional_qm_resids, int):
            self.additional_qm_resids = [self.additional_qm_resids]
        if isinstance(self.additional_qm_resnames, str):
            self.additional_qm_resnames = [self.additional_qm_resnames]

        self._additional_qm_resids = set(self.additional_qm_resids or [])
        for resname in (self.additional_qm_resnames or []):
            self._additional_qm_resids.update(
                self.universe.select_atoms(f"resname {resname}").resids
            )

        if self.custom_qm_region is not None:
            if not isinstance(self.custom_qm_region, dict):
                message = (
                    f"custom_qm_region must be a dict, got {
                        type(self.custom_qm_region).__name__
                    }"
                )
                log.error(message)
                raise TypeError(message)
            missing = {
                "whole_residues", "atom_ids"
            } - self.custom_qm_region.keys()
            if missing:
                message = (
                    f"custom_qm_region is missing required keys: {missing}"
                )
                log.error(message)
                raise ValueError(message)
            whole_residues = self.custom_qm_region["whole_residues"]
            if isinstance(whole_residues, int):
                self.custom_qm_region["whole_residues"] = [whole_residues]
            elif isinstance(whole_residues, list):
                if not all(
                    isinstance(residue, int) for residue in whole_residues
                ):
                    message = (
                        "custom_qm_region['whole_residues'] must be an "
                        "int or list of int"
                    )
                    log.error(message)
                    raise TypeError(message)
            else:
                message = (
                    "custom_qm_region['whole_residues'] must be an "
                    "int or list of int"
                )
                log.error(message)
                raise TypeError(message)
            if not all(
                isinstance(atoms, str)
                for atoms in self.custom_qm_region["atom_ids"]
            ):
                message = (
                    "custom_qm_region['atom_ids'] must be a list of str"
                )
                log.error(message)
                raise TypeError(message)

            self.qm_region = self.custom_qm_region
        else:
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
        metal_resids_for_distance_restraints: Optional[
            Union[int, list[int]]
        ] = None,
        additional_qm_resids: Optional[Union[int, list[int]]] = None,
        additional_qm_resnames: Optional[Union[str, list[str]]] = None,
        custom_qm_region: Optional[dict[str, list]] = None,
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
            metal_resids_for_distance_restraints=(
                metal_resids_for_distance_restraints
            ),
            additional_qm_resids=additional_qm_resids,
            additional_qm_resnames=additional_qm_resnames,
            recipe=recipe,
            custom_qm_region=custom_qm_region
        )

    def _define_qm_region(
            self,
            resids_to_exclude: Optional[Union[int, list[int]]] = None
    ) -> dict[str, list]:
        """Get a simple QM region

        Returns:
            dict[str, list]: QM region split into a list
                             of whole residues and atom ids
        """
        exclude_resids = set(self.exclude_resids or [])
        resids_to_exclude = (
            resids_to_exclude or self.metal_resids_for_distance_restraints
        )
        if resids_to_exclude is not None:
            if isinstance(resids_to_exclude, int):
                resids_to_exclude = [resids_to_exclude]
            exclude_resids.update(resids_to_exclude)

        excluded_atoms = set()
        for metal_atom_idx, metal_ligands in self.coordinating_residues.items(

        ):
            metal_resid = self.universe.select_atoms(
                f"id {metal_atom_idx}"
            ).resids[0]
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

        for resid in getattr(self, "_additional_qm_resids", set()):
            if resid in exclude_resids:
                continue
            residue_atoms = self.universe.select_atoms(f"resid {resid}")
            if not len(residue_atoms):
                continue
            residue = residue_atoms.residues[0]
            if residue in protein.residues:
                qm_region_atom_ids.add(self._get_side_chain_selection(residue))
            else:
                qm_region_whole_residues.add(resid)

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

    def _write_qm_namelist(
            self,
            qm_theory: Optional[str] = "DFTB3",
            qm_shake: Optional[int] = 0
    ):
        parsed_whole_residues = _residue_restraint_mask(
            self.qm_region["whole_residues"]
        )
        atom_ids = ",".join(list(map(str, self.qm_region["atom_ids"])))
        qm_mask = f"':{parsed_whole_residues}|(@{atom_ids})'"

        qm_config_options = {
            "qmmask": str(qm_mask),
            "writepdb": "1",
            "qmcharge": str(self.qm_charge),
            "qm_theory": f"'{qm_theory}'",
            "qmshake": str(qm_shake),
            "qm_ewald": "1",
            "qm_pme": "1"
        }
        qm_namelist = [
            f"  {key}={value}" for key, value in qm_config_options.items()
        ]
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
        protocol: "bssProtocol",
        system: Optional[bssSystem] = None,
        process_name: Optional[str] = "qm-meze-run",
        config_options: Optional[dict] = None,
        qm_theory: str = "DFTB3",
        metal_resids_for_distance_restraints: Optional[
            Union[int, list[int]]
        ] = None,
        is_gpu: bool = False,
        additional_positional_restraints: Optional[dict[str, Any]] = None,
        additional_distance_restraints: Optional[
            dict[tuple[int, int], tuple[float, float, float]]
        ] = None,
        qm_shake: Optional[bool] = False
    ) -> "QuantumMeze":
        if additional_positional_restraints is not None:
            if not (
                {"resids"} <= additional_positional_restraints.keys()
                and not {"resnames"} <= additional_positional_restraints.keys()
            ):
                raise ValueError(
                    "additional_restraints must contain 'resids' "
                    "or 'resnames' keys."
                )

        if not additional_positional_restraints:
            disres = (
                metal_resids_for_distance_restraints
                or self.metal_resids_for_distance_restraints
            )
        else:
            additional_resids = additional_positional_restraints.get(
                "resids", []
            )
            if isinstance(additional_resids, int):
                additional_resids = [additional_resids]
            additional_resids = set(additional_resids)

            additional_resnames = additional_positional_restraints.get(
                "resnames", []
            )
            if isinstance(additional_resnames, str):
                additional_resnames = [additional_resnames]
            additional_resnames = set(additional_resnames)
            for resname in additional_resnames:
                resname_resids = set(
                    atom.resid for atom in self.universe.select_atoms(
                        f"resname {resname}"
                    )
                )
                additional_resids.update(resname_resids)
            additional_resids = list(additional_resids)
            disres = additional_resids

        qm_namelist = self._write_qm_namelist(
            qm_theory=qm_theory, qm_shake=int(qm_shake)
        )

        if disres is not None:
            distance_restraints = self._prepare_distance_restraints(disres)
        else:
            distance_restraints = self.distance_restraints

        if distance_restraints or additional_distance_restraints:
            config_options["nmropt"] = 1
            restraint_namelist = ["&wt TYPE='DUMPFREQ', istep1=1 /"]
        else:
            restraint_namelist = []

        namelist = qm_namelist + restraint_namelist
        extra_distance_restraints = _write_distance_restraints(
            additional_distance_restraints
        ) if additional_distance_restraints else []
        distance_restraints = (
            distance_restraints or []
        ) + extra_distance_restraints
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
    exclude_resids: Optional[Union[int, list[int]]] = field(
        default_factory=list
    )
    metal_resids_for_distance_restraints: Optional[
        Union[int, list[int]]
    ] = None

    @classmethod
    def from_files(
        cls,
        topology: Optional[str] = None,
        coordinates: Optional[str] = None,
        exclude_resids: Optional[Union[int, list[int]]] = [],
        recipe: Optional[Union[dict, "ColdMezeRecipe"]] = None,
        disulfide_bridges: Optional[List[dict[str, int]]] = None,
        metal_resids_for_distance_restraints: Optional[
            Union[int, list[int]]
        ] = None,
        additional_qm_resids: Optional[Union[int, list[int]]] = None,
        additional_qm_resnames: Optional[Union[str, list[str]]] = None,
        custom_qm_region: Optional[dict[str, list]] = None,
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
            message = (
                "Expected 'recipe' to be a ColdMezeRecipe, "
                f"dict, or None, but got {type(recipe).__name__}"
            )
            log.error(message)
            raise TypeError(message)
        return cls(
            topology=topology,
            coordinates=coordinates,
            exclude_resids=exclude_resids,
            metal_resids_for_distance_restraints=(
                metal_resids_for_distance_restraints
            ),
            additional_qm_resids=additional_qm_resids,
            additional_qm_resnames=additional_qm_resnames,
            disulfide_bridges=disulfide_bridges,
            recipe=recipe,
            custom_qm_region=custom_qm_region
        )

    def run(
            self,
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
            timestep: Optional[Union[float, "bssTime"]] = None,
            runtime: Optional[Union[float, "bssTime"]] = None,
            temperature: Optional[Union[float, "bssTemperature"]] = None,
            start_temperature: Optional[Union[float, "bssTemperature"]] = 300,
            end_temperature: Optional[Union[float, "bssTemperature"]] = 300,
            pressure: Optional[Union[float, "bssPressure"]] = None,
            engine_executable: Optional[str] = None,
            qm_theory: Optional[str] = "DFTB3",
            metal_resids_for_distance_restraints: Optional[
                Union[int, list[int]]
            ] = None,
            additional_positional_restraints: Optional[dict[str, Any]] = None,
            additional_distance_restraints: Optional[
                dict[tuple[int, int], tuple[float, float, float]]
            ] = None,
            qm_shake: Optional[bool] = False
    ) -> "ColdQuantumMeze":
        allowed_protocols = ["minimisation", "nvt", "npt"]
        if protocol_type not in allowed_protocols:
            raise ValueError(
                f"Unsupported protocol type '{protocol_type}'.\n"
                f"Must be one of {allowed_protocols}."
            )

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
            start_temperature=(
                start_temperature or self.recipe.start_temperature
            ),
            end_temperature=end_temperature or self.recipe.end_temperature,
            pressure=pressure or self.recipe.pressure,
            path_to_engine=engine_executable or self.recipe.path_to_engine
        )

        config_options = {
            "cut": recipe.nb_cutoff._value,
            "ntpr": 50,
            "ntwx": 50,
            "ntwx": 50,
            "iwrap": 0,
            "ifqnt": 1
        }

        if not qm_shake and recipe.dt._value > 1.0:
            message = (
                "Cannot run a QM/MM MD simulation with a "
                "timestep larger than 1.0 fs without 'qm_shake' set to False."
            )
            log.error(message)
            raise RuntimeError(message)
        elif qm_shake and recipe.dt._value > 1.0:
            message = (
                f"QM shake is on, and the timestep is {recipe.dt} fs,"
                "which is not recommended for QM/MM equilibration."
            )
            log.warning(message)
        else:
            config_options["ntc"] = 1
            config_options["ntf"] = 1

        if restart:
            config_options["irest"] = 1
            config_options["ntx"] = 5

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
                temperature = recipe.temperature

            protocol = bss.Protocol.Equilibration(
                timestep=recipe.dt,
                runtime=recipe.runtime,
                temperature_start=recipe.start_temperature,
                temperature_end=recipe.end_temperature,
                temperature=temperature,
                pressure=None,
            )
        else:
            config_options["barostat"] = recipe.barostat
            protocol = bss.Protocol.Equilibration(
                timestep=recipe.dt,
                runtime=recipe.runtime,
                temperature=recipe.temperature,
                pressure=recipe.pressure,
            )

        return super().run_qm(
            protocol=protocol,
            recipe=recipe,
            system=system,
            process_name=process_name,
            qm_theory=qm_theory,
            metal_resids_for_distance_restraints=(
                metal_resids_for_distance_restraints
            ),
            config_options=config_options,
            is_gpu=False,
            additional_positional_restraints=additional_positional_restraints,
            additional_distance_restraints=additional_distance_restraints,
            qm_shake=qm_shake
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
            metal_resids_for_distance_restraints: Optional[
                Union[int, list[int]]
            ] = None,
            additional_positional_restraints: Optional[dict[str, Any]] = None,
            additional_distance_restraints: Optional[
                dict[tuple[int, int], tuple[float, float, float]]
            ] = None,
    ) -> "ColdQuantumMeze":
        disres = (
            metal_resids_for_distance_restraints
            or self.metal_resids_for_distance_restraints
        )

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
            additional_positional_restraints=additional_positional_restraints,
            additional_distance_restraints=additional_distance_restraints,
        )

    def heat(
            self,
            system: Optional[bssSystem] = None,
            workdir: Optional[str] = None,
            restart: Optional[bool] = False,
            timestep: Optional[Union[float, "bssTemperature"]] = 0.001,
            runtime: Optional[Union[float, "bssTime"]] = None,
            temperature: Optional[Union[float, "bssTemperature"]] = None,
            start_temperature: Optional[Union[float, "bssTemperature"]] = 300,
            end_temperature: Optional[Union[float, "bssTemperature"]] = 300,
            process_name: Optional[str] = "qm-nvt",
            engine_executable: Optional[str] = None,
            qm_theory: Optional[str] = "DFTB3",
            metal_resids_for_distance_restraints: Optional[
                Union[int, list[int]]
            ] = None,
            additional_positional_restraints: Optional[dict[str, Any]] = None,
            additional_distance_restraints: Optional[
                dict[tuple[int, int], tuple[float, float, float]]
            ] = None,
    ) -> "ColdQuantumMeze":
        disres = (
            metal_resids_for_distance_restraints or
            self.metal_resids_for_distance_restraints
        )
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
            additional_positional_restraints=additional_positional_restraints,
            additional_distance_restraints=additional_distance_restraints,
        )

    def pressurise(
            self,
            system: Optional[bssSystem] = None,
            workdir: Optional[str] = None,
            restart: Optional[bool] = False,
            timestep: Optional[Union[float, "bssTemperature"]] = 0.001,
            runtime: Optional[Union[float, "bssTime"]] = None,
            temperature: Optional[Union[float, "bssTemperature"]] = 300,
            pressure: Optional[Union[float, "bssPressure"]] = 1.0,
            process_name: Optional[str] = "qm-npt",
            engine_executable: Optional[str] = None,
            qm_theory: Optional[str] = "DFTB3",
            metal_resids_for_distance_restraints: Optional[
                Union[int, list[int]]
            ] = None,
            additional_positional_restraints: Optional[dict[str, Any]] = None,
            additional_distance_restraints: Optional[
                dict[tuple[int, int], tuple[float, float, float]]
            ] = None,
    ) -> "ColdQuantumMeze":
        disres = (
            metal_resids_for_distance_restraints or
            self.metal_resids_for_distance_restraints
        )
        return self.run(
            protocol_type="npt",
            system=system,
            workdir=workdir,
            restart=restart,
            process_name=process_name,
            timestep=timestep,
            temperature=temperature,
            runtime=runtime,
            pressure=pressure,
            engine_executable=engine_executable,
            qm_theory=qm_theory,
            metal_resids_for_distance_restraints=disres,
            additional_positional_restraints=additional_positional_restraints,
            additional_distance_restraints=additional_distance_restraints,
        )


@dataclass
class HotQuantumMeze(QuantumMeze):
    recipe: HotMezeRecipe
    exclude_resids: Optional[Union[int, list[int]]] = field(
        default_factory=list
    )
    metal_resids_for_distance_restraints: Optional[
        Union[int, list[int]]
    ] = None

    @classmethod
    def from_files(
        cls,
        topology: Optional[str] = None,
        coordinates: Optional[str] = None,
        exclude_resids: Optional[Union[int, list[int]]] = [],
        recipe: Optional[Union[dict, "HotMezeRecipe"]] = None,
        disulfide_bridges: Optional[List[dict[str, int]]] = None,
        metal_resids_for_distance_restraints: Optional[
            Union[int, list[int]]
        ] = None,
        additional_qm_resids: Optional[Union[int, list[int]]] = None,
        additional_qm_resnames: Optional[Union[str, list[str]]] = None,
        custom_qm_region: Optional[dict[str, list]] = None,
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
            message = (
                "Expected 'recipe' to be a HotMezeRecipe, dict,"
                f" or None, but got {type(recipe).__name__}"
            )
            log.error(message)
            raise TypeError(message)
        return cls(
            topology=topology,
            coordinates=coordinates,
            exclude_resids=exclude_resids,
            metal_resids_for_distance_restraints=(
                metal_resids_for_distance_restraints
            ),
            additional_qm_resids=additional_qm_resids,
            additional_qm_resnames=additional_qm_resnames,
            recipe=recipe,
            disulfide_bridges=disulfide_bridges,
            custom_qm_region=custom_qm_region
        )

    def run(
            self,
            workdir: Optional[str],
            system: Optional[bssSystem] = None,
            process_name: Optional[str] = "qm-meze-run",
            ensemble: Optional[Literal["nvt", "npt"]] = "nvt",
            nb_cutoff: Optional[float] = None,
            timestep: Optional[Union[float, "bssTime"]] = 0.001,
            runtime: Optional[Union[float, "bssTime"]] = None,
            temperature: Optional[Union[float, "bssTemperature"]] = 300,
            pressure: Optional[Union[float, "bssPressure"]] = None,
            engine_executable: Optional[str] = None,
            write_frequency: Optional[int] = 500,
            qm_theory: Optional[str] = "DFTB3",
            metal_resids_for_distance_restraints: Optional[
                Union[int, list[int]]
            ] = None,
            additional_positional_restraints: Optional[dict[str, Any]] = None,
            additional_distance_restraints: Optional[
                dict[tuple[int, int], tuple[float, float, float]]
            ] = None,
    ) -> "HotQuantumMeze":
        disres = (
            metal_resids_for_distance_restraints or
            self.metal_resids_for_distance_restraints
        )

        recipe = HotMezeRecipe(
            workdir=workdir or self.recipe.workdir,
            nb_cutoff=nb_cutoff or self.recipe.nb_cutoff,
            runtime=runtime or self.recipe.runtime,
            dt=timestep or self.recipe.dt,
            temperature=temperature or self.recipe.temperature,
            pressure=pressure or self.recipe.pressure,
            path_to_engine=engine_executable or self.recipe.path_to_engine
        )

        config_options = {
            "cut": recipe.nb_cutoff._value,
            "ntpr": write_frequency,
            "ntwx": write_frequency,
            "ntwx": write_frequency,
            "iwrap": 0,
            "irest": 1,
            "ntx": 5,
            "ifqnt": 1
        }
        if ensemble == "npt":
            config_options["barostat"] = 2

        protocol = bss.Protocol.Production(
            timestep=recipe.dt,
            runtime=recipe.runtime,
            temperature=recipe.temperature,
            pressure=recipe.pressure
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
            additional_positional_restraints=additional_positional_restraints,
            additional_distance_restraints=additional_distance_restraints
        )


@dataclass
class Sofra:
    mezes: dict[str, Meze]
    sofra_file: str
    sofra_contents: dict = field(default_factory=dict)
    transformations: Optional[list] = field(default=None)
    lomap_scores: Optional[list] = field(default=None)
    network_file: Optional[str] = field(default=None)
    project_directory: str = field(default_factory=os.getcwd)
    group_name: str = "meze"

    def __str__(self) -> str:
        return _pretty(self)

    @classmethod
    def from_file(
        cls,
        sofra_file: str,
        directory: Optional[str] = None,
        group_name: Optional[str] = None
    ) -> "Sofra":
        if not os.path.isfile(sofra_file):
            message = f"Sofra file not found: {sofra_file}."
            log.error(message)
            raise FileNotFoundError(message)
        with open(sofra_file, "r") as file:
            sofra_contents = json.load(file)
        mezes = {}
        for ligand_name, entry in sofra_contents.items():
            if not isinstance(entry, dict):
                if ligand_name == "network_file":
                    network_file = entry
                    if not os.path.isfile(network_file):
                        message = (
                            "Could not find network file "
                            f"{network_file} from sofra file:\n{sofra_file}"
                        )
                        log.error(message)
                        raise FileNotFoundError(message)
                continue
            try:
                mezes[ligand_name] = Meze.load(entry["pickle_file"])
            except KeyError:
                log.error(f"Could not find pickle file for {ligand_name}: ")
        if not mezes:
            message = (
                f"No pickle mezes found in {sofra_file}. "
                "Cannot construct Sofra object."
            )
            log.error(message)
            raise RuntimeError(message)
        if len(mezes) == 1:
            message = (
                f"Found only one meze in {sofra_file}."
                " Are you sure you wish to continue?"
            )
            log.warning(message)
        if directory:
            if not os.path.isdir(directory):
                message = f"Project directory {directory} does not exist."
                log.error(message)
                raise FileNotFoundError(message)
            project_directory = directory
        else:
            project_directory = os.getcwd()
            log.info(
                "Project directory not set, "
                "using current working directory: \n"
                f"{project_directory}"
            )

        if group_name:
            sofra_group_name = group_name
        else:
            sofra_group_name = list(mezes.values())[0].recipe.group_name

        log.info(
            f"Using group name {sofra_group_name} for the sofra object"
        )

        return cls(
            mezes=mezes,
            sofra_file=sofra_file,
            sofra_contents=sofra_contents,
            project_directory=project_directory,
            group_name=sofra_group_name,
            network_file=network_file
        )

    def average_charges(
            self,
            new_parameterisation_directories: List[str]
    ) -> dict[str, dict[str, float]]:
        if len(new_parameterisation_directories) != len(self.mezes):
            raise ValueError(
                f"Expected {len(self.mezes)} directories,"
                f" got {len(new_parameterisation_directories)}."
            )
        all_charges = {}
        resname_list = []
        for meze in self.mezes.values():

            for metal, residue in meze.coordinating_residues.items():
                resname_list.extend(residue.resnames)

                metal_ag = meze.universe.select_atoms(f"id {metal}")
                if not metal_ag:
                    log.warning(
                        f"Metal atom with id {metal} not found in universe."
                        " Skipping."
                    )
                else:
                    resname_list.extend(metal_ag.resnames)

            resname_list = [
                resname for resname in resname_list
                if resname != meze.ligand_resname
                and resname != meze.ligand.residue_name
            ]

        resname_list = list(set(resname_list))
        all_charges = {resname: {} for resname in resname_list}

        for i, (ligand_name, meze) in enumerate(self.mezes.items()):

            log.info(f"Processing ligand: {ligand_name}")
            parameterisation_directory = new_parameterisation_directories[i]

            for resname, charge_list in all_charges.items():
                atoms = meze.universe.select_atoms(f"resname {resname}")
                if not atoms:
                    log.warning(
                        f"No atoms found with resname {resname} in universe. "
                        "Skipping."
                    )
                    continue
                charge_list = atoms.charges
                if not charge_list.any():
                    log.warning(
                        f"No charges found for resname {resname} in universe. "
                        "Skipping."
                    )
                    continue

                all_charges[resname][ligand_name] = {
                    atom.name: charge
                    for atom, charge in zip(atoms, charge_list)
                }

            charges_file = os.path.join(
                parameterisation_directory, f"{ligand_name}_raw_charges.json"
            )
            with open(charges_file, "w") as f:
                json.dump({"all_charges": all_charges}, f, indent=2)
            log.info(
                f"Raw charges for {ligand_name} written to: {charges_file}"
            )

        averaged_charges = {}

        for resname, ligand_dict in all_charges.items():
            atom_names = list(next(iter(ligand_dict.values())).keys())
            averaged_charges[resname] = {
                atom_name: np.mean(
                    [charges[atom_name] for charges in ligand_dict.values()]
                )
                for atom_name in atom_names
            }

        for resname, atom_dict in averaged_charges.items():
            log.info(f"{resname}:")
            for atom_name, charge in atom_dict.items():
                log.info(f"  {atom_name:6s}  {charge:+.6f}")

        return averaged_charges

    def build_average_charges(
            self, directory: Optional[str] = None
    ) -> list[Meze]:

        new_parameterisation_directories = []
        for meze in self.mezes.values():
            ligand_directory = directory
            if not ligand_directory:
                log.warning(
                    f"parent directory not set, inferring from {
                        meze.parameterisation_directory
                    }"
                )
                ligand_directory = str(
                    pathlib.Path(meze.parameterisation_directory).parent
                )

            parameterisation_directory = os.path.join(
                ligand_directory, "03_averaged_charges"
            )
            parameterisation_directory = parameterisation_directory
            log.info(f"Creating directory: {parameterisation_directory}")
            os.makedirs(parameterisation_directory, exist_ok=True)
            new_parameterisation_directories.append(parameterisation_directory)

        averaged_charges = self.average_charges(
            new_parameterisation_directories
        )
        updated_mezes = []
        for i, (ligand_name, meze) in enumerate(self.mezes.items()):

            log.info(f"Processing ligand: {ligand_name}")

            all_residue_names = []

            for metal, residue in meze.coordinating_residues.items():
                all_residue_names.extend(residue.resnames)

                metal_ag = meze.universe.select_atoms(f"id {metal}")
                if not metal_ag:
                    log.warning(
                        f"Metal atom with id {metal} not found in universe. "
                        "Skipping."
                    )
                else:
                    all_residue_names.extend(metal_ag.resnames)

            residue_names = list(set(
                [resname for resname in all_residue_names
                 if resname != meze.ligand_resname
                 and resname != meze.ligand.residue_name]
            ))

            old_mol2_files = [
                os.path.join(
                    meze.parameterisation_directory, f"{resname}.mol2"
                ) for resname in residue_names
            ]

            new_mol2_files = [
                os.path.join(
                    new_parameterisation_directories[i], f"{resname}.mol2"
                ) for resname in residue_names
            ]

            for j, old_file in enumerate(old_mol2_files):
                if not os.path.isfile((old_file)):
                    log.warning(
                        f"Mol2 file not found: {old_file}. Skipping."
                    )
                    continue
                with open(old_file, "r") as f:
                    data = f.readlines()
                start_index = 0
                for k, line in enumerate(data):
                    if line.startswith("@<TRIPOS>ATOM"):
                        start_index = k + 1
                    if line.startswith("@<TRIPOS>BOND"):
                        end_index = k
                        break

                atoms = data[start_index:end_index]

                new_atoms = []
                for k, atom_line in enumerate(atoms):

                    atom_name = atom_line.split()[1]
                    old_charge = atom_line.split()[8]
                    resname = atom_line.split()[7]

                    new_charge = averaged_charges[resname][atom_name]

                    if not new_charge:
                        log.warning(
                            "Could not find new charge for "
                            f"{resname},{atom_name}. Skipping."
                        )
                        continue

                    formatted_charge = f"{new_charge:.6f}"
                    new_atom_line = atom_line.replace(
                        old_charge, formatted_charge
                    )

                    log.info(
                        f"Replaced {resname}-{atom_name} charge: "
                        f"{old_charge} changed to {new_charge}"
                    )
                    new_atoms.append(new_atom_line)

                with open(new_mol2_files[j], "w") as f:
                    f.writelines(
                        data[:start_index] + new_atoms + data[end_index:]
                    )
                    log.info(
                        f"New charges written to: "
                        f"{new_mol2_files[j]}"
                    )

            if meze.ligand_resname:
                ligand_resname = meze.ligand_resname
            else:
                ligand_resname = meze.ligand.residue_name

            ligand_mol2 = glob.glob(
                f"{meze.parameterisation_directory}/{ligand_resname}.*"
            )

            frcmod_files = glob.glob(
                f"{meze.parameterisation_directory}/*.frcmod"
            )
            mcpbp_pdb_files = glob.glob(
                f"{meze.parameterisation_directory}/*_mcpbpy.pdb"
            )
            tleap_input_files = glob.glob(
                f"{meze.parameterisation_directory}/*_tleap*.in"
            )

            restraint_files = [
                meze.restraint_file
            ] if meze.restraint_file and os.path.isfile(
                meze.restraint_file
            ) else []

            files_to_copy = (
                ligand_mol2 + frcmod_files +
                mcpbp_pdb_files + restraint_files + tleap_input_files
            )

            new_restraints = meze.restraint_file
            tleap_input_file = tleap_input_files[0]
            for old_file in files_to_copy:
                file = os.path.basename(old_file)
                new_file = os.path.join(
                    new_parameterisation_directories[i], file
                )
                try:
                    shutil.copy(old_file, new_file)
                except shutil.SameFileError as e:
                    log.warning(e)
                    log.info(f"Keeping {old_file}")

                if ".RST" in os.path.splitext(file):
                    new_restraints = new_file
                if "tleap" in file:
                    tleap_input_file = new_file

            for input_file in glob.glob(
                f"{new_parameterisation_directories[i]}/*.in"
            ):
                with open(input_file, "r") as f:
                    contents = f.read()
                if meze.parameterisation_directory in contents:
                    updated = contents.replace(
                        meze.parameterisation_directory,
                        new_parameterisation_directories[i]
                    )
                    with open(input_file, "w") as f:
                        f.write(updated)

            new_non_standard_residues = []
            for residue in meze.non_standard_residues:
                if residue.residue_name in residue_names:
                    new_mol2 = os.path.join(
                        new_parameterisation_directories[i],
                        f"{residue.residue_name}.mol2"
                    )
                    if not os.path.isfile(new_mol2):
                        log.warning(
                            f"Expected averaged mol2 not found: {new_mol2}. "
                            "Keeping original."
                        )
                        new_non_standard_residues.append(residue)
                        continue
                    frcmod_src = residue.frcmod_file
                    if frcmod_src and os.path.isfile(frcmod_src):
                        frcmod_dst = os.path.join(
                            new_parameterisation_directories[i],
                            os.path.basename(frcmod_src)
                        )
                    else:
                        frcmod_dst = residue.frcmod_file
                    new_non_standard_residues.append(dataclasses.replace(
                        residue,
                        file=[new_mol2],
                        charge=_get_mol2_charge(new_mol2),
                        frcmod_file=frcmod_dst
                    ))
                else:
                    new_non_standard_residues.append(residue)

            old_ligand = meze.ligand
            new_ligand_mol2 = os.path.join(
                new_parameterisation_directories[i],
                os.path.basename(old_ligand.file[0])
            )
            if not os.path.isfile(new_ligand_mol2):
                message = f"Expected ligand mol2 not found: {new_ligand_mol2}."
                log.error(message)
                raise FileNotFoundError(message)
            new_frcmod = os.path.join(
                new_parameterisation_directories[i],
                os.path.basename(old_ligand.frcmod_file)
            )
            if not os.path.isfile(new_frcmod):
                message = f"Expected ligand frcmod not found: {new_frcmod}."
                log.error(message)
                raise FileNotFoundError(message)
            new_ligand = Ligand(
                file=[new_ligand_mol2],
                charge=old_ligand.charge,
                parameterised=True,
                frcmod_file=new_frcmod,
                residue_name=old_ligand.residue_name
            )

            updated_mezes.append(dataclasses.replace(
                meze,
                non_standard_residues=new_non_standard_residues,
                parameterisation_directory=new_parameterisation_directories[i],
                restraint_file=new_restraints,
                tleap_input_file=tleap_input_file,
                ligand=new_ligand,
                ligand_resname=new_ligand.residue_name
            ))

        solvated_mezes = []
        for ligand_name, updated_meze in zip(
            self.mezes.keys(), updated_mezes
        ):
            solvated = updated_meze.add_water(
                directory=updated_meze.parameterisation_directory,
                mcpbpy_tleap_file=updated_meze.tleap_input_file
            )
            pickle_file = solvated.save(
                filename=os.path.join(
                    solvated.parameterisation_directory,
                    f"{ligand_name}_avg_charges_solv"
                )
            )

            solvated.add_to_sofra(
                key=ligand_name,
                filename=self.sofra_file,
                pickle_file=pickle_file,
            )
            self.mezes[ligand_name] = solvated
            self.sofra_contents[ligand_name]["parameterisation_directory"] = (
                solvated.parameterisation_directory
            )
            self.sofra_contents[ligand_name]["pickle_file"] = pickle_file
            if solvated.tleap_input_file:
                self.sofra_contents[ligand_name]["tleap_input_file"] = (
                    solvated.tleap_input_file
                )
            if solvated.restraint_file:
                self.sofra_contents[ligand_name]["restraint_file"] = (
                    solvated.restraint_file
                )
            if os.path.isfile(solvated.topology):
                self.sofra_contents[ligand_name]["topology"] = (
                    solvated.topology
                )
            if os.path.isfile(solvated.coordinates):
                self.sofra_contents[ligand_name]["coordinates"] = (
                    solvated.coordinates
                )

            solvated_mezes.append(solvated)

        with open(self.sofra_file, "w") as f:
            json.dump(self.sofra_contents, f, indent=4)

        return solvated_mezes

    def set_ligand_network(
            self,
            pdb_files: list[str],
            directory: Optional[str] = None,
            force_connected_ligands_file: Optional[str] = None
    ):
        # TODO
        # have an option to input a network to allow users
        # to use the same network accross models
        if not pdb_files:
            message = "Could not find any pdb files in the given path"
            log.error(message)
            raise RuntimeError(message)

        sdf_files = pdb_to_sdf(pdb_files)

        if directory:
            lomap_directory = os.path.join(directory, "lomap")
        else:
            lomap_directory = "lomap"

        os.makedirs(lomap_directory, exist_ok=True)
        log.info(
            f"Created lomap directory at: \n"
            f"{lomap_directory}"
        )
        for sdf_file in sdf_files:
            shutil.copy(sdf_file, lomap_directory)

        os.chdir(lomap_directory)
        lomap_command = f"lomap -d -n {self.group_name} "
        if force_connected_ligands_file:
            if not os.path.isfile(force_connected_ligands_file):
                log.warning(
                    "Could not find links file: "
                    f"{force_connected_ligands_file}\n"
                    "Continuing without it."
                )
            lomap_command += f"-l {force_connected_ligands_file} "

        lomap_command += ". "
        log.info(
            f"Running lomap with command: \n"
            f"{lomap_command}"
        )

        lomap_run_result = subprocess.run(
            lomap_command,
            shell=True,
            capture_output=True,
            text=True
        )

        if lomap_run_result.returncode != 0:
            log.warning(
                "Lomap exited with a non-zero exit code "
                f"{lomap_run_result.returncode}:"
            )
            log.warning(lomap_run_result.stderr)
            log.warning(
                "It's likely that the network was still "
                "generated succesfully, checking... "
            )

        scores_file = os.path.join(
            lomap_directory, f"{self.group_name}_score_with_connection.txt"
        )
        png_file = os.path.join(lomap_directory, f"{self.group_name}.png")
        if not os.path.isfile(scores_file):
            message = (
                f"Lomap did not produce {scores_file}. "
                "Check lomap output for errors."
            )
            log.error(message)
            raise RuntimeError(message)
        if not os.path.isfile(png_file):
            message = (
                f"Lomap did not produce {png_file}. "
                "Check lomap output for errors."
            )
            log.error(message)
            raise RuntimeError(message)

        log.info("Lomap finished succesfully. Parsing outputs.")
        self.transformations, self.lomap_scores, network_file = (
            self._parse_lomap_output(scores_file, lomap_directory)
        )
        self.save_network_file(network_file)

    def save_network_file(self, network_file: str):
        self.network_file = network_file
        self.sofra_contents["network_file"] = network_file
        with open(self.sofra_file, "w") as file:
            json.dump(self.sofra_contents, file, indent=4)
        log.info(
            f"Saved network file path to {self.sofra_file}:\n{network_file}"
        )

    def _parse_lomap_output(self, file: str, directory: str):
        transformations, scores = [], []
        cleaned_rows = []
        row_i = 0
        with open(file, "r") as ifile:
            reader = csv.DictReader(ifile)
            reader.fieldnames = [key.strip() for key in reader.fieldnames]
            for row in reader:
                name_1 = pathlib.Path(row["Filename_1"].strip()).stem
                name_2 = pathlib.Path(row["Filename_2"].strip()).stem
                score = float(row["Str_sim"].strip())
                connect = pathlib.Path(row["Connect"].strip()).stem

                if connect.lower() == "yes":
                    if row_i == 0:
                        cleaned_rows.append("Name_1,Name_2,Score\n")
                    transformations.append((name_1, name_2))
                    scores.append(score)
                    clean_row = f"{name_1},{name_2},{score}\n"
                    cleaned_rows.append(clean_row)
                    row_i += 1

        if not cleaned_rows:
            message = (
                "Lomap output did not contain any connected ligands. "
                "Check lomap outputs for any errors."
            )
            log.error(message)
            raise RuntimeError(message)

        connected_file = os.path.join(
            directory, f"{self.group_name}_lomap_network.csv"
        )
        with open(connected_file, "w") as ofile:
            ofile.writelines(cleaned_rows)
        log.info(f"Wrote lomap network to file:\n{connected_file}")

        return transformations, scores, connected_file


@dataclass
class AlchemicalSofra:
    first_meze: Meze
    second_meze: Meze
    stage: Literal["bound", "unbound"]
    recipe: AlchemicalMezeRecipe
    first_name: Optional[str] = "ligand_1"
    second_name: Optional[str] = "ligand_2"
    system_sofra: Optional[Sofra] = field(default=None)
    directory: Optional[str] = field(default=None)
    overwrite: bool = field(default=False)
    bss_base_system: Optional[bssSystem] = field(default=None, init=False)
    first_molecule: Optional[bss._SireWrappers.Molecule] = field(
        default=None, init=False
    )
    second_molecule: Optional[bss._SireWrappers.Molecule] = field(
        default=None, init=False
    )
    merged_molecule: Optional[bss._SireWrappers.Molecule] = field(
        default=None, init=False
    )
    working_directory: Optional[str] = field(default=None, init=False)
    transformation: Optional[str] = field(default=None, init=False)

    def __post_init__(self):
        if self.stage not in ["bound", "unbound"]:
            message = f"stage must be 'bound' or 'unbound', got {self.stage}"
            log.error(message)
            raise ValueError(message)

        if self.recipe.engine.upper() not in ["SOMD"]:
            message = "Currently only supporting SOMD as the RBFE MD engine."
            log.error(message)
            raise RuntimeError(message)

        self._set_bss_molecules()
        self._set_bss_base_system()
        self._set_transformation()
        self._set_working_directory()

    def _set_working_directory(self):

        directory = self.directory or os.getcwd()

        if not os.path.isdir(directory):
            message = f"Input directory {directory} does not exist."
            log.warning(message)
            os.makedirs(directory)

        working_directory = os.path.join(
            directory, self.transformation, self.stage
        )
        log.info(
            f"Creating {self.stage} stage in directory: {working_directory}"
        )
        try:
            os.makedirs(working_directory, exist_ok=self.overwrite)
        except OSError:
            message = (
                f"Directory {working_directory} already exists. "
                "Set overwrite=True or supply a different directory."
            )
            log.error(message)
            raise FileExistsError(message)
        self.working_directory = working_directory

    def _set_bss_molecules(self):
        self.first_molecule = self.first_meze.get_mutatable_ligand_molecule()
        self.second_molecule = self.second_meze.get_mutatable_ligand_molecule()

    def _set_bss_base_system(self):
        self.bss_base_system = self.first_meze.system
        log.info(
            "Setting base system from the first meze object."
        )

    def _set_network(self, file: str):
        self.network = pd.read_csv(file, sep=",", header=0, index_col=False)
        log.info(f"Read in network:\n{self.network.head()}")

    def _set_transformation(self):
        self.transformation = f"{self.first_name}~{self.second_name}"
        log.info(f"Setting up transformation: {self.transformation}")

    def merge(
            self,
            flexible_align: bool = False,
            ring_breaks: bool = True,
            ring_size_changes: bool = True
    ):
        mapping = bss.Align.matchAtoms(
            self.first_molecule, self.second_molecule, complete_rings_only=True
        )
        inverse_mapping = {value: key for key, value in mapping.items()}
        if flexible_align:
            aligned_ligand_2 = bss.Align.flexAlign(
                self.second_molecule, self.first_molecule, inverse_mapping
            )
        else:
            aligned_ligand_2 = bss.Align.rmsdAlign(
                self.second_molecule, self.first_molecule, inverse_mapping
            )
        return bss.Align.merge(
            self.first_molecule,
            aligned_ligand_2,
            mapping,
            allow_ring_breaking=ring_breaks,
            allow_ring_size_change=ring_size_changes
        )

    def create_hybrid_molecule(self):
        self.merged_molecule = self.merge(
            self.recipe.flexible_align,
            self.recipe.ring_breaks,
            self.recipe.ring_size_changes
        )
        system = self.bss_base_system
        system.removeMolecules(self.first_molecule)
        system.addMolecules(self.merged_molecule)
        return system

    def setup_alchemistry(
            self,
            compute_platform: Literal["cuda", "opencl", "cpu"] = "cuda",
            n_somd_cycles: Optional[int] = None,
            n_somd_moves: Optional[int] = None,
            n_frames: Optional[int] = 250,
            buffered_coordinates_frequency: Optional[int] = None,
            only_save_end_states: bool = False,
            debug: bool = False
    ):

        merged_ligand_system = self.create_hybrid_molecule()

        n_somd_cycles = n_somd_cycles or int(
            self.recipe.sampling_time._value * 5
        )
        n_somd_moves = n_somd_moves or _set_n_somd_moves(
            sampling_time=self.recipe.sampling_time._value,
            n_somd_cycles=n_somd_cycles,
            stepsize=self.recipe.dt._value
        )
        buffered_coordinates_frequency = buffered_coordinates_frequency or max(
            int(n_somd_moves / n_frames), 10000
        )
        n_cycles_per_saved_frame = max(
            1, self.recipe.restart_interval // n_somd_moves
        )

        config_options = {
            "ncycles": n_somd_cycles,
            "nmoves": n_somd_moves,
            "buffered coordinates frequency": buffered_coordinates_frequency,
            "ncycles_per_snap": n_cycles_per_saved_frame,
            "minimal coordinate saving": only_save_end_states,
            "minimise": self.recipe.minimise_lambda,
            "minimise maximum iterations": (
                self.recipe.lambda_minimisation_steps
            ),
            "cutoff distance": self.recipe.nb_cutoff,
            "platform": compute_platform,
            "verbose": debug
        }

        if self.stage == "bound" and self.first_meze.restraint_file:
            somd_restraints = _write_somd_restraints(
                self.first_meze.restraint_file
            )
            config_options["use permanent distance restraints"] = True
            config_options["permanent distance restraints dictionary"] = (
                somd_restraints
            )

        elif self.stage == "bound" and self.recipe.model == 0:
            log.warning(
                "Model 0 bound stage requested without a restraint_file "
                f"on {self.first_name}; "
                "no metal-coordinating distance restraints will be applied."
            )

        free_energy_protocol = bss.Protocol.FreeEnergy(
            num_lam=self.recipe.n_lambdas,
            runtime=self.recipe.sampling_time,
            timestep=self.recipe.dt,
            temperature=self.recipe.temperature,
            pressure=self.recipe.pressure,
            restart_interval=self.recipe.restart_interval,
            report_interval=self.recipe.report_interval
        )

        bss.FreeEnergy.Relative(
            system=merged_ligand_system,
            protocol=free_energy_protocol,
            work_dir=self.working_directory,
            engine=self.recipe.engine,
            setup_only=True,
            extra_options=config_options
        )

        generated_configurations = glob.glob(
            os.path.join(self.working_directory, "*", "*.cfg")
        )

        _remove_gpu_from_fep_configs(generated_configurations)

        return merged_ligand_system
