from meze import ColdQuantumMeze, HotQuantumMeze
import os
import sys

system = "mII"

project_dir = "/Users/af25016/projects/meze/data/"

input_dir = f"{project_dir}/equilibration/{system}"

cold_qm_meze = ColdQuantumMeze.from_files(
    topology=f"{input_dir}/06_relax/next.prm7",
    coordinates=f"{input_dir}/06_relax/next.rst7",
    group_name=f"qm_mII_{system}",
    path_to_engine=os.path.join(os.environ["AMBERHOME"], "bin", "sander"),
    additional_qm_resnames=["0YB", "ROH", "4YA", "0MA", "2MA", "VMA", "VMB"]
)

qmmm_dir = os.path.join(project_dir, "qmmm", system)
os.makedirs(qmmm_dir, exist_ok=True)

k = 60
radius = 1
restraints ={
    (16186, 16187): (0.096, k, radius), #Atom names: HO1-O1; Atom types: Ho-Oh
}


print("Minimising")

minimised_qm_meze = cold_qm_meze.minimise(
    process_name="01_qm_min",
    workdir=qmmm_dir,
    max_cycles=10, 
    additional_restraints=restraints
)

print("Heating")

minimised_qm_meze.heat(
    process_name="02_qm_heat",
    workdir=qmmm_dir,
    timestep=0.001,
    runtime=50,
    start_temperature=100,
    end_temperature=300,
    additional_restraints=restraints
) 

print("Production")

hot_qm_meze = HotQuantumMeze.from_files(
    topology=f"{qmmm_dir}/02_qm_heat/next.prm7",
    coordinates=f"{qmmm_dir}/02_qm_heat/next.rst7",
    group_name=f"qm_vim2_{system}",
    path_to_engine=os.path.join(os.environ["AMBERHOME"], "bin", "sander")
)

hot_qm_meze.run(
    process_name="03_qm_prod",
    workdir=qmmm_dir,
    additional_restraints=restraints
)