from meze import ColdMeze
import os

project_dir = "/Users/af25016/projects/indole-carboxylates/vim2/"
input_dir = f"{project_dir}/system_preparation/water"
cold_meze = ColdMeze.from_files(topology=f"{input_dir}/vim2_solv.prmtop",
                                coordinates=f"{input_dir}/vim2_solv.inpcrd",
                                group_name="vim2_wat")

equil_dir = os.path.join(project_dir, "equilibration")
os.makedirs(equil_dir, exist_ok=True)

cold_meze.minimise(workdir=equil_dir,
                   position_restraints="solute",
                   restraint_weight=100.0)

