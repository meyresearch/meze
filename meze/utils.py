import numpy as np 
from .ligand import Ligand
from typing import (
    List,
    Optional
)
import os

def residue_restraint_mask(residue_ids: list[int]) -> str:
    """Generate an Amber-style restraint mask.

    Adapted from: 
    https://github.com/OpenBioSim/biosimspace/blob/devel/python/BioSimSpace/_Config/_amber.py#L203
    
    Args:
        residue_ids (list[int]): list of residue ids that should be restrained

    Returns:
        str: amber-style residue id-based restraint mask
    """
    residue_ids = list(set(residue_ids))
    residue_ids.sort()

    if len(residue_ids) == 1:
        restraint_mask = f"{residue_ids[0]}"
    else:
        restraint_mask = f"{residue_ids[0]}"
        previous_id = residue_ids[0]
        lead_id = previous_id

        for idx in residue_ids[1:]:
            if idx - previous_id > 1:
                if previous_id != lead_id:
                    restraint_mask += f"{previous_id},{idx}"
                else:
                    restraint_mask += f",{idx}"
                lead_id = idx
            else:
                if idx - lead_id == 1:
                    restraint_mask += "-"
            previous_id = idx
        if idx - residue_ids[-2] == 1:
            restraint_mask += f"{idx}"
        else:
            if idx != lead_id:
                restraint_mask += f",{idx}"

    return restraint_mask

def write_distance_restraints(
        restraints: dict[tuple[int, int], tuple[float, float, float]]
) -> list[str]:
    lines = []

    for (metal_id, ligating_atom_id), (distance, force_constant, flat_bottom_radius) in restraints.items():
        
        r1 = np.round(distance - flat_bottom_radius, 2)
        r2 = np.round(distance - flat_bottom_radius / 2, 2)
        r3 = np.round(distance + flat_bottom_radius / 2, 2)
        r4 = np.round(distance + flat_bottom_radius, 2)

        line = (
            f"&rst iat={metal_id},{ligating_atom_id}, "
            f"r1={r1}, r2={r2}, r3={r3}, r4={r4}, "
            f"rk2={force_constant}, rk3={force_constant}, /\n"
        )
        lines.append(line)
    return lines

def write_tleap_solvation_input(protein_file: str,
                                ligand: Ligand,
                                non_standard_residues: Optional[List[Ligand]] = None,
                                disulfide_bridges: Optional[List[dict[str, int]]] = None,
                                workdir: Optional[str] = "", #TODO move the below to model recipe: 
                                protein_ff: Optional[str] = "ff14SB",
                                water_model: Optional[str] = "tip3p",
                                box_shape: Optional[str] = "octahedral",
                                box_edges: Optional[float] = 10.0,
                                solvent_closeness: Optional[float] = 0.75,
                                ligand_ff: Optional[str] = "gaff2",
                                ):
    if workdir:
        os.chdir(workdir)
    lines = [
        f"source oldff/leaprc.{protein_ff}\n",
        f"source leaprc.water.{water_model.lower()}\n",       
    ]
    if "tip3p" in water_model.lower():
        lines.append(
            "loadamberparams frcmod.ions1lm_126_tip3p\n"
        )

    lines.extend([
        f"source leaprc.{ligand_ff}\n",
        f"loadamberparams {ligand.frcmod_file}\n",
        f"lig = loadmol2 {ligand.file[0]}\n",
        f"\n"
    ])
    if non_standard_residues:
        res_names = []
        for _, res in enumerate(non_standard_residues, start=1):
            var = res.residue_name 
            res_names.append(var)

            lines.extend([
                f"loadamberparams {res.frcmod_file}\n",
                f"{var} = loadmol2 {res.file[0]}\n",
                "\n"
            ])
    lines.append(
        f"protein = loadpdb {protein_file}\n"
    )

    if disulfide_bridges:

        for bridge in disulfide_bridges:
            resid1 = bridge["resid1"]
            resid2 = bridge["resid2"]
            lines.append(
                f"bond protein.{resid1}.SG protein.{resid2}.SG\n"
            )
        lines.append("\n")

    lines.extend([
        "complex = combine {protein lig}\n",
        f"savepdb complex {ligand.name}_complex_dry.pdb\n",
        f"check complex\n"
        "\n"
    ])
        
    if "oct" in box_shape.lower():
        lines.append(
            f"solvate{box_shape[:3]} complex {water_model.upper()}BOX {box_edges} iso {solvent_closeness}\n"
        )
    else:
        lines.append(
            f"solvate{box_shape[:3]} complex {water_model.upper()}BOX {box_edges} {solvent_closeness}\n"
        )

    lines.extend([
        "addions2 complex Na+ 0\n",
        "addions2 complex Cl- 0\n",
        "\n"
        f"savepdb complex {ligand.name}_complex_solv.pdb\n",
        f"saveamberparm complex {ligand.name}_complex_solv.prmtop {ligand.name}_complex_solv.inpcrd\n",
        "quit"
    ])
    return lines


