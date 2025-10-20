from meze import ColdMeze, HotMeze
import os

system_name = "vim2"
bridging_water = "hydroxide"

project_dir = f"/Users/af25016/projects/indole-carboxylates/{system_name}/"
input_dir = f"{project_dir}/equilibration/{bridging_water}"

cold_meze = ColdMeze.from_files(
    topology=f"{input_dir}/06_relax/next.prm7",
    coordinates=f"{input_dir}/06_relax/next.rst7",
    group_name=f"{system_name}_{bridging_water}",
    path_to_executable=os.path.join(
        os.environ["AMBERHOME"], "bin", "sander"
    ),
    model="0"
)

# NPT equil no position restraints, add distance+angle restraints
hot_meze = cold_meze.pressurise(
    workdir=input_dir,
    restart=True,
    process_name="07_free",
    is_gpu=False
)



