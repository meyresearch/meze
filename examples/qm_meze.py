from meze import ColdQuantumMeze, HotQuantumMeze
import os
import sys

system = sys.argv[1]

project_dir = "/user/work/af25016/indole-carboxylates/"

input_dir = f"{project_dir}/equilibration/{system}"

cold_qm_meze = ColdQuantumMeze.from_files(
    topology=f"{input_dir}/06_relax/next.prm7",
    coordinates=f"{input_dir}/06_relax/next.rst7",
    group_name=f"qm_vim2_{system}",
    path_to_engine=os.path.join(os.environ["AMBERHOME"], "bin", "sander")
)

qmmm_dir = os.path.join(project_dir, "qmmm", system)
os.makedirs(qmmm_dir, exist_ok=True)


print("Minimising")

minimised_qm_meze = cold_qm_meze.minimise(
    process_name="01_qm_min",
    workdir=qmmm_dir,
    max_cycles=10000
)

print("Heating")

minimised_qm_meze.heat(
    process_name="02_qm_heat",
    workdir=qmmm_dir,
    timestep=0.001,
    runtime=50,
    start_temperature=100,
    end_temperature=300
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
    workdir=qmmm_dir
)