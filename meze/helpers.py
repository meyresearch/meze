import shutil
import os
import logging
from rich.logging import RichHandler
from rich.console import Console
console = Console(force_terminal=True, color_system="truecolor")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)],
    force=True,
)

log = logging.getLogger("rich")


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

        message = (
            f"{tool} not found. AmberTools installation required."
        )
        log.error(message)
        raise RuntimeError(message)
