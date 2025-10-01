from meze import ColdMeze

project_dir = "/Users/af25016/projects/indole-carboxylates/vim2/system_preparation/water"
cold_meze = ColdMeze.from_files(topology=f"{project_dir}/vim2_solv.prmtop",
                                coordinates=f"{project_dir}/vim2_solv.inpcrd",
                                group_name="vim2_wat")


cold_meze.minimise(position_restraints="all",
                   workdir=project_dir)

