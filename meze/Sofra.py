from dataclasses import (
    dataclass,
    field
)
from meze import Protein
from typing import (
    List,
    Optional
)

@dataclass
class Sofra:
    protein_file: str
    group_name: str = "meze"
    protein: Optional[Protein] = None 
    ligands: List = field(default_factory=list)
