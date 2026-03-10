import shutil
import os

def _check_ambertools():
    required = [
        "antechamber",
        "parmchk2", 
        "tleap", 
        "MCPB.py", 
        "metalpdb2mol2.py", 
        "pdb4amber"
    ]

    for tool in required:
        if shutil.which(tool):
            continue

        amberhome = os.environ.get("AMBERHOME")
        if amberhome and os.path.exists(os.path.join(amberhome, "bin", tool)):
            continue

        raise RuntimeError(f"{tool} not found. AmberTools installation required.")