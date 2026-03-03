from meze import ColdQuantumMeze, HotQuantumMeze
import os
import sys

system = "mII"

project_dir = "/Users/af25016/projects/meze/data/"

cold_qm_meze = ColdQuantumMeze.from_files(
    topology=f"{project_dir}/inputs/qmmm/protein/{system}/complex.prm7",
    coordinates=f"{project_dir}/inputs/qmmm/protein/{system}/complex.rst7",
    group_name="qm_mII"
)

qmmm_dir = f"{project_dir}/outputs/qmmm/{system}/"
os.makedirs(qmmm_dir, exist_ok=True)

print("Minimising")

minimised_qm_meze = cold_qm_meze.minimise(
    process_name="01_qm_min",
    workdir=qmmm_dir,
    max_cycles=100
)