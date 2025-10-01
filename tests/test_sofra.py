from meze import Sofra, Meze

project_dir = "/Users/af25016/projects/indole-carboxylates/vim2/system_preparation/water"
meze_ = Meze.from_files(topology=f"{project_dir}/vim2_solv.prmtop",
                        coordinates=f"{project_dir}/vim2_solv.inpcrd",
                        group_name="vim2_wat")
print(meze_)

active_site = meze_.get_active_site()
for atom in active_site:
    print(atom.resname, atom.resid)