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
            raise TypeError(f"Expected str or list[str], got {type(self.file)}")

        for file in self.file:
            if not os.path.isfile(file):
                raise FileNotFoundError(f"Ligand file not found: {file}")

        if not isinstance(self.charge, float):
            try:
                self.charge = float(self.charge)
            except (TypeError, ValueError):
                raise TypeError(
                    f"Ligand charge must be an integer or float (got {self.charge} of type {type(self.charge)})."
                )
        if not self.name:
            self.name = Path(self.file[0]).stem
            warnings.warn(
                f"Ligand name not set, inferring from file name: {self.file}",
                UserWarning
            )
        
        self.system = bss.IO.readMolecules(self.file)

    def parameterise(self, 
                     directory: Optional[str] = None,
                     method: Literal["antechamber", "tleap"] = "antechamber",  
                     atom_type: str = "gaff2",
                     charge_method: str = "bcc", 
                     residue_name: str = "MOL", 
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
        old_resname = [line.split()[3] for line in lines if "HETATM" in line][0]
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
            pass

        return dataclasses.replace(
            self,
            file=output_coordinate_file,
            parameterised=True,
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
        workdir = os.getcwd()
        lines = [f"source leaprc.{force_field}\n",
                 f"loadpdb {coordinate_file}\n",
                 f"savemol2 {residue_name}.mol2 {atom_type}\n",
                 "quit"]
        with open(f"{parameterisation_directory}/{residue_name}_tleap.in", "w") as ofile: 
            ofile.writelines(lines)
        tleap_cmd = f"tleap -s -f {parameterisation_directory}/{residue_name}_tleap.in"
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
