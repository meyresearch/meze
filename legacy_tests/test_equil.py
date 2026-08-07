from meze import ColdMeze
import os

project_dir = "/Users/af25016/projects/indole-carboxylates/vim2/"
input_dir = f"{project_dir}/system_preparation/water"

cold_meze = ColdMeze.from_files(
    topology=f"{input_dir}/vim2_solv.prmtop",
    coordinates=f"{input_dir}/vim2_solv.inpcrd",
    group_name="vim2_wat",
    path_to_engine=os.path.join(os.environ["AMBERHOME"], "bin", "sander")
)


equil_dir = os.path.join(project_dir, "equilibration")
os.makedirs(equil_dir, exist_ok=True)

minimised_meze = cold_meze.minimise(
    process_name="01_min",
    workdir=equil_dir,
    position_restraints="solute",
    max_cycles=100, 
    is_gpu=True
)

hot_meze = cold_meze.heat(
    system=minimised_meze.system,
    process_name="02_heat",
    workdir=equil_dir,
    position_restraints="solute",
    timestep=0.001,
    runtime=10,
    start_temperature=100,
    end_temperature=300
) 

relax_meze = cold_meze.heat(
    system=hot_meze.system,
    restart=True,
    process_name="03_relax",
    workdir=equil_dir,
    runtime=10,
    position_restraints="solute",
    timestep=0.001,
)

pressure_meze = cold_meze.pressurise(
    system=relax_meze.system,
    restart=True,
    process_name="04_pressure",
    workdir=equil_dir, 
    position_restraints="solute",
    timestep=0.001,
)

lower_restraint = cold_meze.pressurise(
    system=pressure_meze.system,
    restart=True,
    process_name="05_lower",
    workdir=equil_dir,
    position_restraints="solute",
    timestep=0.001,
    restraint_weight=10.0
)

relax_backbone = cold_meze.pressurise(
    system=lower_restraint.system,
    restart=True,
    process_name="06_relax",
    workdir=equil_dir,
    position_restraints="backbone",
    timestep=0.001,
    restraint_weight=10.0
)

