import BioSimSpace as bss
from BioSimSpace._SireWrappers import System as bssSystem
from pathlib import Path
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
    parameterised: bool = False

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

    def parameterise(self, # NEED TO ADD CORRECT PATH! OTHERWISE IT'S IN THE CWD
                     atom_type: str = "gaff2",
                     charge_method: str = "bcc", 
                     residue_name: str = "MOL"):
        
        if len(self.file) > 1:
            raise UserWarning(f"Expected one ligand file but got {self.file}")
        else:
            file = self.file[0]
        
        ext = os.path.splitext(file)

        antechamber_cmd = (
            f"antechamber -fi {ext} -fo mol2 "
            f"-i {file} -o {self.name}.mol2 "
            f"-nc {charge_method} -c {self.charge} -at {atom_type} "
            f"-pf y -rn {residue_name}"
        )
        print("Running antechamber with command:")
        print(antechamber_cmd)
        os.system(antechamber_cmd)
        parmcheck_cmd = (
            f"parmchk2 -i {self.name}.mol2 -o {self.name}.frcmod "
            f"-f mol2 -s {atom_type}"
        )
        print("Running parmchk2 with command:")
        print(parmcheck_cmd)
        os.system(parmcheck_cmd)
