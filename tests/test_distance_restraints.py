from meze import ColdQuantumMeze, HotQuantumMeze
import os

system_name = "l1"
bridging_water = "water"

topology_filename = "L153_1SML_HOH_wat.parm7"
coordinate_filename = "L153_1SML_HOH_wat.rst7"

project_dir = f"/Users/af25016/projects/indole-carboxylates/{system_name}/"
input_dir = f"{project_dir}/system_preparation/{bridging_water}"

cold_qm_meze = ColdQuantumMeze.from_files(
    topology=f"{input_dir}/{topology_filename}",
    coordinates=f"{input_dir}/{coordinate_filename}",
    group_name=f"{system_name}_{bridging_water}",
    path_to_executable=os.path.join(
        os.environ["AMBERHOME"], "bin", "sander"
    ),
    exclude_resids=800,
    metal_resids_for_distance_restraints=[536, 537]
)

qmmm_dir = os.path.join(project_dir, "qmmm", "water")
os.makedirs(qmmm_dir, exist_ok=True)

minimised_qm_meze = cold_qm_meze.minimise(
    process_name="01_qm_min",
    workdir=qmmm_dir,
    max_cycles=5
)

heated_qm_meze = minimised_qm_meze.heat(
    process_name="02_qm_heat",
    workdir=qmmm_dir,
    runtime=5,
    start_temperature=100,
    end_temperature=300
) 

hot_qm_meze = HotQuantumMeze.from_files(
    topology=f"{input_dir}/{topology_filename}",
    coordinates=f"{input_dir}/{coordinate_filename}",
    group_name=f"{system_name}_{bridging_water}",
    path_to_executable=os.path.join(
        os.environ["AMBERHOME"], "bin", "sander"
    ),
    exclude_resids=800,
    metal_resids_for_distance_restraints=[536, 537]
)

qmmm_dir = os.path.join(project_dir, "qmmm", bridging_water)
os.makedirs(qmmm_dir, exist_ok=True)


hot_qm_meze.run(
    workdir=qmmm_dir,
    process_name="03_qmmm_prod",
    runtime=5
)