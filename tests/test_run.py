from meze import HotMeze
import os
import sys

repeat = sys.argv[1]

project_dir = "/Users/af25016/projects/indole-carboxylates/vim2/"
equilibration_dir = f"{project_dir}/system_preparation/water"

meze = HotMeze.from_files(
    topology=f"{equilibration_dir}/next.prm7",
    coordinates=f"{equilibration_dir}/next.rst7",
    group_name="vim2_wat",
    path_to_engine=os.path.join(os.environ["AMBERHOME"], "bin", "sander")
)

outputs_dir = os.path.join(project_dir, "outputs", "water", f"repeat_{repeat+1}")
os.makedirs(outputs_dir, exist_ok=True)


meze.run(
    process_name="test_prod",
    workdir=outputs_dir,
    runtime=0.5
)
