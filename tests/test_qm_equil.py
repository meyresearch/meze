from meze import ColdQuantumMeze
import os

project_dir = "/Users/af25016/projects/indole-carboxylates/vim2/"
equilibration_dir = f"{project_dir}/equilibration/water"

cold_qm_meze = ColdQuantumMeze.from_files(
    topology=f"{equilibration_dir}/06_relax/next.prm7",
    coordinates=f"{equilibration_dir}/06_relax/next.rst7",
    group_name="qm_vim2",
    path_to_engine=os.path.join(os.environ["AMBERHOME"], "bin", "sander")
)

qmmm_dir = os.path.join(project_dir, "qmmm", "water")
os.makedirs(qmmm_dir, exist_ok=True)

minimised_qm_meze = cold_qm_meze.minimise(
    process_name="01_qm_min",
    workdir=qmmm_dir,
    max_cycles=10
)

hot_meze = minimised_qm_meze.heat(
    process_name="02_qm_heat",
    workdir=qmmm_dir,
    timestep=0.001,
    runtime=10,
    start_temperature=100,
    end_temperature=300
) 