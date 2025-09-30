from meze import Sofra, Meze

project_dir = "/Users/af25016/projects/indole-carboxylates/vim2/system_preparation/water"
sofra = Sofra(protein_file=f"{project_dir}/vim2_solv.pdb")
print(sofra)

meze_ = Meze(protein_file=f"{project_dir}/vim2_solv.pdb")
print(meze_)