from dataclasses import dataclass
import Protein
from typing import (
    List,
    Optional
)

@dataclass
class Sofra:
    protein_file: str
    group_name: str = "meze"
    protein: Optional[Protein.Protein] = None 
    ligands: List = []
