import sys
import pathlib

ligand_directory = sys.argv[1]
ligand = ligand_directory.split("/")[-2]

script_templates = ["slurm_g_opt.sh",
                    "slurm_mk.sh"]

for script in script_templates:
    step = pathlib.Path(script).stem

    with open(script, "r") as ifile:
        input_lines = ifile.readlines()

    fixed_lines = []
    for line in input_lines:
        if f"--job-name=g-opt" in line:
            line = line.replace("g-opt", f"{ligand}")
        elif f"--job-name=g-mk" in line:
            line = line.replace("g-mk", f"{ligand}")
        if f"vim2_large_opt.com" in line:
            line = line.replace(f"vim2_large_opt.com", f"vim2_{ligand}_large_opt.com")
        elif f"vim2_large_mk.com" in line:
            line = line.replace(f"vim2_large_mk.com", f"vim2_{ligand}_large_mk.com")
        fixed_lines.append(line)

    ligand_step_file = f"{ligand_directory}/{ligand}_{script}"
    with open(ligand_step_file, "w") as ofile:
        ofile.writelines(fixed_lines)
