from meze import ColdQuantumMeze, HotQuantumMeze
import os
import sys
from pathlib import Path

REPO_ROOT = Path().resolve()  
EXAMPLES_DIR = REPO_ROOT / "examples"
project_dir = REPO_ROOT / "data"

system = "mII"

input_dir = f"{project_dir}/equilibration/model_0/{system}/m5g0/"

cold_qm_meze = ColdQuantumMeze.from_files(
    topology=f"{input_dir}/next.prm7",
    coordinates=f"{input_dir}/next.rst7",
    group_name=f"qm_mII_{system}",
    path_to_engine=os.path.join(os.environ["AMBERHOME"], "bin", "sander"),
    additional_qm_resnames=["0YB", "ROH", "4YA", "0MA", "2MA", "VMA", "VMB"],
    custom_qm_region={
        "whole_residues": [1015, 1022],
        "atom_ids": [
            "16186",
            "16248-16251",
            "2837-2842",
            "1002-1007",
            "7207-7217",
            "971-981"
        ]
    }
)

qmmm_dir = os.path.join(project_dir, "outputs", "qmmm", system)
os.makedirs(qmmm_dir, exist_ok=True)

print("Minimising")

minimised_qm_meze = cold_qm_meze.minimise(
    process_name="01_qm_min",
    workdir=qmmm_dir,
    max_cycles=10, 
)

print("Equilibrating")

minimised_qm_meze.pressurise(
    process_name="02_qm_npt",
    workdir=qmmm_dir,
    timestep=0.001,
    runtime=5,
    temperature=300,
    pressure=1.0,
    restart=True
) 

print("Production")

hot_qm_meze = HotQuantumMeze.from_files(
    topology=f"{qmmm_dir}/02_qm_npt/next.prm7",
    coordinates=f"{qmmm_dir}/02_qm_npt/next.rst7",
    group_name=f"qm_vim2_{system}",
    path_to_engine=os.path.join(os.environ["AMBERHOME"], "bin", "sander"),
    custom_qm_region={
        "whole_residues": [1015, 1022],
        "atom_ids": [
            "16186",
            "16248-16251",
            "2837-2842",
            "1002-1007",
            "7207-7217",
            "971-981"
        ]
    }
)

hot_qm_meze.run(
    process_name="03_qm_prod",
    workdir=qmmm_dir,
    ensemble="npt",
    pressure=1.0,
    temperature=300
)