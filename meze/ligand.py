import BioSimSpace as bss
from BioSimSpace._SireWrappers import System as bssSystem
from pathlib import Path
import dataclasses
from dataclasses import dataclass
import warnings
from typing import (
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

        if not isinstance(self.charge, int):
            try:
                self.charge = int(self.charge)
            except (TypeError, ValueError):
                raise TypeError(
                    f"Ligand charge must be an integer (got {self.charge} of type {type(self.charge)})."
                )
        if not self.name:
            self.name = Path(self.file[0]).stem
            warnings.warn(
                f"Ligand name not set, inferring from file name: {self.file}",
                UserWarning
            )
        
        self.system = bss.IO.readMolecules(self.file)

    def parameterise(self, 
                     path: str | None = None,
                     atom_type: str = "gaff2",
                     charge_method: str = "bcc", 
                     residue_name: str = "MOL"):
        
        if len(self.file) > 1:
            raise UserWarning(f"Expected one ligand file but got {self.file}")
        else:
            file = self.file[0]
        
        with open(file, "r") as ifile:
            lines = ifile.readlines()
        old_resname = [line.split()[3] for line in lines][0]
        new_lines = [line.replace(old_resname, residue_name) for line in lines]
        
        with open(f"{path}/{residue_name}.pdb", "w") as ofile:
            ofile.writelines(new_lines)

        file = f"{path}/{residue_name}.pdb"

        ext = Path(file).suffix[1:]

        os.makedirs(path, exist_ok=True)

        mol2_path = os.path.join(path, f"{self.name}.mol2")
        workdir = os.getcwd()
        antechamber_cmd = (
            f"antechamber -fi {ext} -fo mol2 "
            f"-i {file} -o {mol2_path} "
            f"-c {charge_method} -nc {self.charge} -at {atom_type} "
            f"-pf y -rn {residue_name}"
        )
        print("Running antechamber with command:")
        print(antechamber_cmd)
        os.chdir(path)
        os.system(antechamber_cmd)
        if not os.path.isfile(mol2_path): 
            warnings.warn(
                f"antechamber failed: missing output files for {self.name}.mol2",
                UserWarning
            )
        
        with open(mol2_path, "r") as ifile:
            mol2_lines = ifile.readlines()
        new_lines = []
        for line in mol2_lines:
            if "DU" in line:
                warnings.warn(f"Atom type DU found in file {mol2_path}")
                atom_name = line.split()[1]
                new_line = line.replace("DU", atom_name)
                warnings.warn(f"Replacing DU with {atom_name}")
            else:
                new_line = line
            new_lines.append(new_line)
        with open(mol2_path, "w") as ofile:
            ofile.writelines(new_lines)
        
        frcmod_path = os.path.join(path, f"{self.name}.frcmod")
        parmcheck_cmd = (
            f"parmchk2 -i {mol2_path} -o {frcmod_path} "
            f"-f mol2 -s {atom_type}"
        )
        print("Running parmchk2 with command:")
        print(parmcheck_cmd)
        os.system(parmcheck_cmd)


        if not os.path.isfile(frcmod_path):
            warnings.warn(
                f"parmchk2 failed: missing output files for {self.name}.frcmod",
                UserWarning
            )
        os.chdir(workdir)

        return dataclasses.replace(
            self,
            file=mol2_path,
            parameterised=True,
            frcmod_file=frcmod_path,
            residue_name=residue_name
        )       
        
