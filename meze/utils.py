import warnings
import numpy as np
from .ligand import Ligand
from typing import (
    List,
    Optional,
    Dict,
    Literal
)
import os
from dataclasses import fields, is_dataclass
from collections.abc import Mapping, Iterable
import glob
from pathlib import Path
from rich.logging import RichHandler
from rich.console import Console
import logging

console = Console(force_terminal=True, color_system="truecolor")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)],
    force=True,
)

log = logging.getLogger("rich")


def _list_rindex(list_to_search: list[str], word: str) -> int:
    """
     Source - https://stackoverflow.com/a/3940144
     Posted by wim, modified by community.
     See post 'Timeline' for change history
     Retrieved 2026-01-20, License - CC BY-SA 4.0
     """
    rindex = None

    for i, item in enumerate(reversed(list_to_search)):
        if word in item:
            rindex = i
            break
    if rindex is None:
        raise ValueError(f"'{word}' is not in list")
    rlist = [line for line in reversed(list_to_search)]
    return [
        i for i, line in enumerate(list_to_search) if line == rlist[rindex]
    ][-1]


def _residue_restraint_mask(residue_ids: list[int]) -> str:
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


def _write_distance_restraints(
        restraints: dict[tuple[int, int], tuple[float, float, float]]
) -> list[str]:
    lines = []

    for (metal_id, ligating_atom_id), (
        distance, force_constant, flat_bottom_radius
    ) in restraints.items():

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


def _remove_gpu_from_fep_configs(files: List[str]) -> None:
    for file in files:
        with open(file, "r") as f:
            lines = [line for line in f if "gpu" not in line.lower()]
        with open(file, "w") as f:
            f.writelines(lines)


def _write_tleap_solvation_input(
        protein_file: str,
        ligand: Ligand,
        non_standard_residues: Optional[List[Ligand]] = None,
        disulfide_bridges: Optional[List[dict[str, int]]] = None,
        workdir: Optional[str] = "",  # TODO move the below to model recipe:
        protein_ff: Optional[str] = "ff14SB",
        water_model: Optional[str] = "tip3p",
        box_shape: Literal["octahedral", "cubic"] = "octahedral",
        box_edges: Optional[float] = 10.0,
        solvent_closeness: Optional[float] = 0.75,
        ligand_ff: Optional[str] = "gaff2"
):
    allowed_shapes = ["octahedral", "cubic"]
    if box_shape.lower() not in allowed_shapes:
        message = (
            f"Box shape '{box_shape}' not allowed. "
            f"Must be one of {allowed_shapes}"
        )
        log.error(message)
        raise ValueError(message)
    if workdir:
        os.chdir(workdir)
    lines = [
        f"source leaprc.protein.{protein_ff}\n",
        f"source leaprc.water.{water_model.lower()}\n",
    ]
    if "tip3p" in water_model.lower():
        lines.append(
            "loadamberparams frcmod.ions1lm_126_tip3p\n"
        )
    if ligand:
        lines.extend([
            f"source leaprc.{ligand_ff}\n",
            f"loadamberparams {ligand.frcmod_file}\n",
            f"lig = loadmol2 {ligand.file[0]}\n",
            "\n"
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
        "check complex\n"
        "\n"
    ])

    if box_shape.lower() == "octahedral":
        lines.append(
            f"solvateOct complex {water_model.upper()}BOX "
            f"{box_edges} iso {solvent_closeness}\n"
        )
    else:
        lines.append(
            f"solvateBox complex {water_model.upper()}BOX "
            f"{box_edges} {solvent_closeness}\n"
        )

    lines.extend([
        "addions2 complex Na+ 0\n",
        "addions2 complex Cl- 0\n",
        "\n"
        f"savepdb complex {ligand.name}_complex_solv.pdb\n",
        f"saveamberparm complex {ligand.name}_complex_solv.prmtop "
        f"{ligand.name}_complex_solv.inpcrd\n",
        "quit"
    ])
    return lines


def _edit_mcpbpy_tleap_input(
        tleap_input_file: str,
        # TODO move the below to model recipe:
        box_shape: Literal["octahedral", "cubic"] = "octahedral",
        box_edges: Optional[float] = 10.0,
        water_model: Optional[str] = "tip3p",
        solvent_closeness: Optional[float] = 0.75,
        ligand_ff: Optional[str] = "gaff2"
):
    if not os.path.isfile(tleap_input_file):
        raise FileNotFoundError(
            f"Could not find tleap input file: "
            f"{tleap_input_file}"
        )
    allowed_shapes = ["octahedral", "cubic"]
    if box_shape.lower() not in allowed_shapes:
        message = (
            f"Box shape '{box_shape}' not allowed. "
            f"Must be one of {allowed_shapes}"
        )
        log.error(message)
        raise ValueError(message)

    with open(tleap_input_file, "r") as ifile:
        tleap_lines = ifile.readlines()

    if not f"source leaprc.{ligand_ff}\n" in tleap_lines:
        tleap_lines.insert(0, f"source leaprc.{ligand_ff}\n")

    old_solvate_line = [
        line for line in tleap_lines if "solvate" in line.lower()
    ][0]
    pdb_line = [line for line in tleap_lines if "loadpdb" in line.lower()][0]
    variable_name = pdb_line.split("=")[0].strip()
    if box_shape.lower() == "octahedral":
        new_solvate_line = (
            f"solvateOct {variable_name} "
            f"{water_model.upper()}BOX {box_edges} iso {solvent_closeness}\n"
        )

    else:
        new_solvate_line = (
            f"solvateBox {variable_name} "
            f"{water_model.upper()}BOX {box_edges} {solvent_closeness}\n"
        )

    tleap_lines = [
        line.replace(
            old_solvate_line, new_solvate_line
        ) for line in tleap_lines
    ]

    return tleap_lines


def _write_gaussian_script(
        job_name: str,
        gaussian_version: str,
        script_name: str,
        com_file: str,
        directory: str,
        sbatch_options: dict[str] = None,
        additional_lines: list[str] = None
) -> str:

    gaussian_script_file = os.path.join(directory, script_name)

    with open(gaussian_script_file, "w") as gscript:
        gscript.write("#!/bin/bash\n")

        gscript.write("\n")
        gscript.write(f"#SBATCH --job-name={job_name}\n")
        if sbatch_options:
            for key, value in sbatch_options.items():
                gscript.write(f"#SBATCH {key}={value}\n")
            gscript.write("\n")

        if additional_lines:
            for line in additional_lines:
                gscript.write(f"{line}\n")
            gscript.write("\n")

        gscript.write(
            f"{gaussian_version} {com_file}"
        )

    return gaussian_script_file


def _pretty(obj, indent=0, step=2):
    pad = " " * indent

    if is_dataclass(obj):
        cls = type(obj).__name__
        parts = []
        for f in fields(obj):
            value = getattr(obj, f.name)
            parts.append(
                f"{pad}{' ' * step}{f.name}="
                f"{_pretty(value, indent + step, step)}"
            )
        inner = ",\n".join(parts)
        return f"{cls}(\n{inner}\n{pad})"

    if isinstance(obj, Mapping):
        parts = [
            f"{pad}{' ' * step}{repr(k)}: {_pretty(v, indent + step, step)}"
            for k, v in obj.items()
        ]
        inner = ",\n".join(parts)
        return "{\n" + inner + "\n" + pad + "}"

    if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        parts = [
            _pretty(v, indent + step, step) for v in obj
        ]
        if len(parts) > 1:
            inner = ",\n".join(f"{pad}{' ' * step}{p}" for p in parts)
            return "[\n" + inner + "\n" + pad + "]"
        else:
            return "[" + ", ".join(parts) + "]"

    return repr(obj)


def _parse_mcpbpy_input(mcpbpy_input_file: str) -> dict:

    if not mcpbpy_input_file:
        raise ValueError("mcpbpy.in file is not set")
    elif not os.path.isfile(mcpbpy_input_file):
        raise FileNotFoundError(
            f"mcpbpy.in file does not exist: "
            f"{mcpbpy_input_file}"
        )
    mcpb_input_options = {}
    with open(mcpbpy_input_file, "r") as file:
        for line in file:
            parts = line.split()
            if len(parts) == 2:
                (key, value) = parts[0], parts[1]
            else:
                (key, value) = parts[0], parts[1:]
            mcpb_input_options[key] = value

    if {"cut_off"} <= mcpb_input_options.keys():
        mcpb_input_options["cut_off"] = float(mcpb_input_options["cut_off"])
    if {"ion_ids"} <= mcpb_input_options.keys():
        value = mcpb_input_options["ion_ids"]
        if isinstance(value, str):
            mcpb_input_options["ion_ids"] = int(value)
        else:
            mcpb_input_options["ion_ids"] = [int(v) for v in value]
    if {"gaff"} <= mcpb_input_options.keys():
        mcpb_input_options["gaff"] = int(mcpb_input_options["gaff"])
    if {"large_opt"} <= mcpb_input_options.keys():
        mcpb_input_options["large_opt"] = int(mcpb_input_options["large_opt"])
    return mcpb_input_options


def _check_log_files(directory: str) -> List[str]:
    log_files = glob.glob(
        f"{directory}/*.log"
    )
    if len(log_files) == 0:
        warnings.warn(
            "Could not find any log files in: "
            f"{directory}",
            UserWarning
        )
    elif len(log_files) == 1:
        warnings.warn(
            "Only one log file found in the parameterisation directory: "
            f"{log_files[0]}",
            UserWarning
        )
    for log_file in log_files:
        if "leap.log" not in log_file:
            with open(log_file, "r") as ifile:
                contents = ifile.read()
            with open(log_file, "r") as ifile:
                lines = ifile.readlines()
            if not lines:
                raise IOError(
                    f"Log file {log_file} is empty."
                )
            if "Normal termination of Gaussian" not in contents:
                warnings.warn(
                    f"Log file {log_file} did not terminate normally",
                    UserWarning
                )
            if "large_opt" in log_file:
                try:
                    i = _list_rindex(lines, "Converged")
                except ValueError as e:
                    message = (
                        str(e) +
                        f"\nLog file {log_file} may not have converged."
                    )
                    log.error(message)
                    raise RuntimeError(message) from e
                convergence_lines = " ".join(lines[i+1:i+5])
                max_force_line = lines[i+1]
                rms_force_line = lines[i+2]
                max_displacement_line = lines[i+3]
                rms_displacement_line = lines[i+4]

                if (
                    "YES" not in max_force_line or
                    "YES" not in rms_force_line or
                    "YES" not in max_displacement_line or
                    "YES" not in rms_displacement_line
                ):
                    warnings.warn(
                        f"Log file {log_file} did not converge:"
                        f"{convergence_lines}",
                        UserWarning
                    )
    return log_files


def _get_mol2_charge(file: str) -> int | float:
    with open(file, "r") as ifile:
        lines = ifile.readlines()

    start = None
    end = None
    for i, line in enumerate(lines):
        if "ATOM" in line:
            start = i + 1
        if "BOND" in line:
            end = i

    if not start and not end:
        message = f"Could not read mol2 file: {file}"
        log.error(message)
        raise RuntimeError(message)

    atom_lines = lines[start:end]
    charges = [float(line.split()[-1].strip()) for line in atom_lines]
    return sum(charges)


def pdb_to_sdf(files: str | list[str]):

    if isinstance(files, str):
        files = [files]

    output_files = []

    for ligand_file in files:
        name = Path(ligand_file).stem
        _, extension = os.path.splitext(ligand_file)
        log.info(f"Read in file: {ligand_file}")
        log.info(f"Using name {name} for output")
        output_file = os.path.join(
            os.path.dirname(ligand_file),
            f"{name}.sdf"
        )
        obabel_command = (
            f"obabel -i {extension.strip('.')} "
            f"{ligand_file} -o sdf -O {output_file}"
        )
        log.info("Converting to sdf with obabel command: \n")
        log.info(obabel_command)
        os.system(obabel_command)
        output_files.append(output_file)

    if not output_files:
        message = "Could not convert files to sdf"
        log.error(message)
        raise RuntimeError(message)

    return output_files


def _set_n_somd_moves(
        sampling_time: float,
        n_somd_cycles: int,
        stepsize: float = 0.002,

) -> int:
    """
    Set SOMD nmoves in a reasonable way for better performance.
    See reference to: https://github.com/michellab/BioSimSpace/issues/258
    and
    https://github.com/OpenBioSim/biosimspace/issues/18
    """
    time = sampling_time * 1000
    number_of_steps = int(time / stepsize)
    return number_of_steps // n_somd_cycles


def _write_somd_restraints(
        amber_style_restraint_file: str
) -> Dict[tuple[int, int], tuple[float, float, float]]:
    if not os.path.isfile(amber_style_restraint_file):
        message = (
            "Could not find AMBER-style restraint file: "
            f"{amber_style_restraint_file}"
        )
        log.error(message)
        raise FileNotFoundError
    somd_restraints = {}
    with open(amber_style_restraint_file, "r") as ifile:
        for line in ifile:
            iat1 = int(line.split("iat=")[1].split(",")[0]) - 1
            iat2 = int(line.split("iat=")[1].split(",")[1]) - 1
            r2 = float(line.split("r2=")[1].split(",")[0])
            r3 = float(line.split("r3=")[1].split(",")[0])
            rk2 = float(line.split("rk2=")[1].split(",")[0])
            flat_bottom_radius = np.round((r3 - r2)/2, decimals=2)
            equilibrium_distance = np.round(
                r2 + flat_bottom_radius, decimals=2
            )
            force_constant = np.round(rk2, decimals=2)

            atom_key = (iat1, iat2)
            restraint_value = (
                equilibrium_distance, force_constant, flat_bottom_radius
            )
            somd_restraints[atom_key] = restraint_value
    return somd_restraints
