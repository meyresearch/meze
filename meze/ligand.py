import logging
import BioSimSpace as bss
from BioSimSpace._SireWrappers import System as bssSystem
from pathlib import Path
import dataclasses
from dataclasses import dataclass
import warnings
from typing import (
    Literal,
    Optional,
    Union,
    List
)
import os
from .helpers import _check_ambertools
from rich.logging import RichHandler
from rich.console import Console
import shutil
console = Console(force_terminal=True, color_system="truecolor")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)],
    force=True,
)

log = logging.getLogger("rich")


@dataclass
class Ligand():
    file: Union[str, List[str]]
    name: Optional[str] = None
    charge: int = 0
    system: Optional[bssSystem] = None
    atom_type: Optional[str] = "gaff2"
    parameterised: bool = False
    frcmod_file: Optional[str] = None
    residue_name: Optional[str] = None
    topology: Optional[str] = None
    coordinates: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.file, str):
            self.file = [self.file]
        elif isinstance(self.file, list):
            if len(self.file) > 2:
                raise ValueError(
                    f"Too many values for 'file': {self.file}."
                    f"Expected a 'str' or a list of at most 2 input files."
                )
            self.file = self.file
        else:
            raise TypeError(
                f"Expected str or list[str], got {type(self.file)}"
            )

        for file in self.file:
            if not os.path.isfile(file):
                raise FileNotFoundError(f"Ligand file not found: {file}")

        if not isinstance(self.charge, float):
            try:
                self.charge = float(self.charge)
            except (TypeError, ValueError):
                raise TypeError(
                    f"Ligand charge must be an integer or float \
                    (got {self.charge} of type {type(self.charge)})."
                )
        if not self.name:
            self.name = Path(self.file[0]).stem
            warnings.warn(
                f"Ligand name not set, inferring from file name: {self.file}",
                UserWarning
            )

        self.system = bss.IO.readMolecules(self.file)

        if not self.residue_name:
            residues = self.system.getResidues()
            if len(residues) > 1:
                log.warning(
                    f"Found multiple residues in ligand file {self.file}: \
                    \n{residues}"
                )
                log.warning("Choosing residue name based on first residue.")
            residue = residues[0]
            self.residue_name = residue.name()
            log.info(f"Ligand residue name set to {self.residue_name}")

    def parameterise(self,
                     directory: Optional[str] = None,
                     method: Literal["antechamber", "tleap"] = "antechamber",
                     atom_type: str = "gaff2",
                     charge_method: str = "bcc",
                     residue_name: str = "MOL",
                     force_field: Optional[str] = None,
                     filename: Optional[str] = None) -> "Ligand":

        if not directory:
            directory = os.getcwd()

        if len(self.file) > 1:
            raise UserWarning(f"Expected one ligand file but got {self.file}")
        else:
            file = self.file[0]

        output_filename = filename or f"{self.name}"

        with open(file, "r") as ifile:
            lines = ifile.readlines()
        old_resname = [
            line.split()[3] for line in lines if "HETATM" in line
            or "ATOM" in line
        ][0]
        new_lines = [line.replace(old_resname, residue_name) for line in lines]

        with open(f"{directory}/{residue_name}.pdb", "w") as ofile:
            ofile.writelines(new_lines)

        file = f"{directory}/{residue_name}.pdb"
        os.makedirs(directory, exist_ok=True)

        output_coordinate_file = None
        output_frcmod_file = None
        if method == "antechamber":
            mol2_path = os.path.join(directory, f"{output_filename}.mol2")
            output_coordinate_file = self._run_antechamber(
                parameterisation_directory=directory,
                input_file=file,
                output_file=mol2_path,
                atom_type=atom_type,
                charge_method=charge_method,
                residue_name=residue_name
            )
            frcmod_path = os.path.join(directory, f"{output_filename}.frcmod")
            output_frcmod_file = self._run_parmchk2(
                parameterisation_directory=directory,
                input_file=mol2_path,
                output_file=frcmod_path,
                atom_type=atom_type
            )
        elif method == "tleap":
            output_coordinate_file = self.run_ligand_tleap(
                parameterisation_directory=directory,
                coordinate_file=file,
                residue_name=residue_name,
                force_field=force_field
            )
        new_system = bss.IO.readMolecules(output_coordinate_file)
        return dataclasses.replace(
            self,
            file=output_coordinate_file,
            parameterised=True,
            system=new_system,
            frcmod_file=output_frcmod_file,
            residue_name=residue_name
        )

    def _run_antechamber(self,
                         parameterisation_directory: str,
                         input_file: str,
                         output_file: str,
                         atom_type: str = "gaff2",
                         charge_method: str = "bcc",
                         residue_name: str = "MOL",):
        _check_ambertools()
        charge = int(self.charge)
        workdir = os.getcwd()
        antechamber_cmd = (
            f"antechamber -fi pdb -fo mol2 "
            f"-i {input_file} -o {output_file} "
            f"-c {charge_method} -nc {charge} -at {atom_type} "
            f"-pf y -rn {residue_name}"
        )
        logging.info("Running antechamber with command:")
        logging.info(antechamber_cmd)
        os.chdir(parameterisation_directory)
        os.system(antechamber_cmd)
        if not os.path.isfile(output_file):
            warnings.warn(
                f"antechamber failed: missing output files for {output_file}",
                UserWarning
            )

        with open(output_file, "r") as ifile:
            mol2_lines = ifile.readlines()
        new_lines = []
        for line in mol2_lines:
            if "DU" in line:
                warnings.warn(f"Atom type DU found in file {output_file}")
                atom_name = line.split()[1]
                new_line = line.replace("DU", atom_name)
                warnings.warn(f"Replacing DU with {atom_name}")
            else:
                new_line = line
            new_lines.append(new_line)
        with open(output_file, "w") as ofile:
            ofile.writelines(new_lines)
        os.chdir(workdir)
        return output_file

    def _run_parmchk2(self,
                      parameterisation_directory: str,
                      input_file: str,
                      output_file: str,
                      atom_type: str = "gaff2"):
        _check_ambertools()
        workdir = os.getcwd()
        parmcheck_cmd = (
            f"parmchk2 -i {input_file} -o {output_file} "
            f"-f mol2 -s {atom_type}"
        )
        logging.info("Running parmchk2 with command:")
        logging.info(parmcheck_cmd)
        os.chdir(parameterisation_directory)
        os.system(parmcheck_cmd)

        if not os.path.isfile(output_file):
            warnings.warn(
                f"parmchk2 failed: missing output files for {output_file}",
                UserWarning
            )
        os.chdir(workdir)
        return output_file

    def run_ligand_tleap(self,
                         parameterisation_directory: str,
                         coordinate_file: str,
                         force_field: str,
                         residue_name: str = "MAN",
                         atom_type: Literal["default", "amber"] = "default"):
        _check_ambertools()
        workdir = os.getcwd()
        if atom_type == "default":
            atom_type = "0"
        else:
            atom_type = "1"
        lines = [f"source leaprc.{force_field}\n",
                 f"lig = loadpdb {coordinate_file}\n",
                 f"savemol2 lig {residue_name}.mol2 {atom_type}\n",
                 "quit"]
        with open(
            f"{parameterisation_directory}/{residue_name}_tleap.in", "w"
        ) as ofile:
            ofile.writelines(lines)
        tleap_cmd = f"tleap -s -f \
            {parameterisation_directory}/{residue_name}_tleap.in"
        logging.info("Running tleap with command:")
        logging.info(tleap_cmd)
        os.chdir(parameterisation_directory)
        os.system(tleap_cmd)
        output_file = f"{parameterisation_directory}/{residue_name}.mol2"
        if not os.path.isfile(output_file):
            warnings.warn(
                f"tleap failed: missing output files for {output_file}",
                UserWarning
            )
        os.chdir(workdir)
        return output_file

    def add_water(
            self,
            directory: Optional[str] = None,
            force_field: Optional[str] = "gaff2",
            water_model: Optional[str] = "tip3p",
            box_shape: Optional[str] = "octahedral",
            box_edges: Optional[float] = 10.0,
            solvent_closeness: Optional[float] = 0.75,
    ):
        _check_ambertools()

        if directory:
            os.makedirs(directory, exist_ok=True)

        ligand = (
            self.parameterise(directory) if not self.parameterised else self
        )

        ligand_mol2 = os.path.join(directory, f"{ligand.name}.mol2")
        ligand_frcmod = os.path.join(directory, f"{ligand.name}.frcmod")
        if not os.path.isfile(ligand_mol2):
            shutil.copy(ligand.file[0], ligand_mol2)
        if not os.path.isfile(ligand_frcmod):
            shutil.copy(ligand.frcmod_file, ligand_frcmod)

        tleap_input_file = os.path.join(directory, "tleap_solvate.in")
        tleap_output_file = os.path.join(directory, "tleap_solvate.out")

        lines = [f"source leaprc.{force_field}\n",
                 f"source leaprc.water.{water_model.lower()}\n"]

        if "tip3p" in water_model.lower():
            lines.append(
                "loadamberparams frcmod.ions1lm_126_tip3p\n"
            )

        lines.extend([
            f"loadamberparams {ligand_frcmod}\n",
            f"lig = loadmol2 {ligand_mol2}\n"
            f"\n"
        ])

        if "oct" in box_shape.lower():
            lines.append(
                f"solvate{box_shape[:3]} lig "
                f"{water_model.upper()}BOX {box_edges} iso "
                f"{solvent_closeness}\n"
            )
        else:
            lines.append(
                f"solvate{box_shape[:3]} lig "
                f"{water_model.upper()}BOX {box_edges} {solvent_closeness}\n"
            )

        lines.extend([
            "addions2 lig Na+ 0\n",
            "addions2 lig Cl- 0\n",
            "\n",
            f"savepdb lig {ligand.name}_solv.pdb\n",
            f"saveamberparm lig {ligand.name}_solv.prmtop "
            f"{ligand.name}_solv.inpcrd\n",
            "quit"
        ])

        with open(tleap_input_file, "w") as ifile:
            ifile.writelines(lines)
        solvated_topology = f"{ligand.name}_solv.prmtop"
        solvated_coordinates = f"{ligand.name}_solv.inpcrd"

        workdir = os.getcwd()
        os.chdir(directory)
        tleap_command = f"tleap -s -f {tleap_input_file} > {tleap_output_file}"
        log.info("Running tleap with command:")
        log.info(tleap_command)
        os.system(tleap_command)

        try:
            solvated_topology = os.path.join(
                directory,
                solvated_topology
            )
            solvated_coordinates = os.path.join(
                directory,
                solvated_coordinates
            )
            system = bss.IO.readMolecules(
                [solvated_topology, solvated_coordinates]
            )

            solvated_ligand = dataclasses.replace(
                ligand,
                file=ligand_mol2,
                frcmod_file=ligand_frcmod,
                topology=solvated_topology,
                coordinates=solvated_coordinates,
                system=system,
                parameterised=True
            )
        except FileNotFoundError:
            log.error(f"Failed to solvate ligand {ligand.name}")
            raise RuntimeError
        os.chdir(workdir)

        return solvated_ligand
