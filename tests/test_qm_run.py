from meze import HotQuantumMeze
import os

project_dir = "/Users/af25016/projects/indole-carboxylates/vim2/"
equilibration_dir = f"{project_dir}/equilibration/water"

hot_qm_meze = HotQuantumMeze.from_files(
    topology=f"{equilibration_dir}/06_relax/next.prm7",
    coordinates=f"{equilibration_dir}/06_relax/next.rst7",
    group_name="qm_vim2",
    path_to_engine=os.path.join(os.environ["AMBERHOME"], "bin", "sander")
)

qmmm_dir = os.path.join(project_dir, "qmmm", "water")
os.makedirs(qmmm_dir, exist_ok=True)

hot_qm_meze.run(
    workdir=qmmm_dir,
    process_name="03_qmmm_prod",
    runtime=5
)