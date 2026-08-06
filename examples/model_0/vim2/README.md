# Model 0 — VIM-2

Distance and angle restraint model applied to the VIM-2 metallo-β-lactamase system.

Each submission script reads `ligands.txt` and dispatches a SLURM array job — one task per ligand. Steps 2, 3, and 4 each run 3 independent repeats.

---

## Pipeline overview

```
01_submit_solvation.sh          # Solvate bound complex (protein + ligand)
02_submit_heat.sh               # Equilibrate bound complex (8-stage protocol)
03_submit_production.sh         # Production MD (bound, model 0 restraints)
04_submit_ligand_solvation_and_heating.sh  # Solvate and equilibrate unbound ligand
scripts/prepare_network.py      # Build ligand transformation network
scripts/run_alchemistry_prep.sh # Submit alchemistry preparation jobs
```

---

## Step-by-step

### Required inputs

Create an inputs directory for your protein and the recipe as a `JSON` file: 
```
<project_dir>/inputs/model_0/protein/vim2/
    vim2.fixed.pdb                       # Prepared protein structure
    model_0_recipe.json                  # Model 0 parameters (restraints, force field options)
```

Create a ligands directory with each ligand in its own directory: 

```
<project_dir>/inputs/model_0/ligands/vim2/
    <ligand>/<ligand>.pdb                # Ligand structures
```

You should also create a `ligands.txt` file that has all the ligand names in a single file. You can do this with: 
```
ls -d */ | tr -d '/' > ligands.txt
```

### Example data

Example data can be found in [`data/inputs/model_0/protein/vim2/`](https://github.com/meyresearch/meze/blob/1923d9c3549e9f75d4f1a3c5ad327c5dd4001bbe/data/inputs/model_0/protein/vim2) and [`data/inputs/model_0/ligands/vim2/ligand_11.pdb`](https://github.com/meyresearch/meze/blob/213463264df740b518616943975278b9b22258ae/data/inputs/model_0/ligands/vim2/ligand_11.pdb)


### 1. Solvate bound complex

```bash
bash 01_submit_solvation.sh <project_dir>
```

`scripts/add_water.py` — loads the protein PDB and `model_0_recipe.json` into a `ColdMeze` object, adds the ligand, solvates the complex, and saves the solvated system as a pickle file. The entry is registered in `model_0_sofra.json` for downstream use.

Output: `<project_dir>/inputs/model_0/protein/vim2/solvate_<ligand>_bound/`

---

### 2. Equilibrate bound complex

```bash
bash 02_submit_heat.sh <project_dir> <system_name>
```

`scripts/heat_meze.py` — runs an 8-stage equilibration on the solvated complex using AMBER `sander`:

| Stage | Process | Restraints | Notes |
|-------|---------|------------|-------|
| 01 | Minimisation | Solute | 5000 cycles |
| 02 | NVT heat 100→300 K | Solute | dt = 0.001 ps |
| 03 | NVT 300 K | Solute | dt = 0.001 ps |
| 04 | NPT | Solute | dt = 0.001 ps |
| 05 | NPT | Solute | Pos. res. weight lowered to 10 kcal/mol/Å² |
| 06 | NPT | Backbone + metal coordination | Pos. res. weight = 10 kcal/mol/Å² |
| 07 | NPT | Metal coordination only | Pos. res. weight = 1 kcal/mol/Å² |
| 08 | NPT | None (free) | Final equilibrated state |

Output: `<project_dir>/equilibration/model_0/vim2/bound/<ligand>/repeat_<N>/`

---

### 3. Production MD (bound)

```bash
bash 03_submit_production.sh <project_dir>
```

`scripts/production.py` — loads a `HotMeze` from the `08_free` equilibration checkpoint and runs 150 ns production MD using `pmemd.cuda` (dt = 0.002 ps). The model 0 distance and angle restraints are applied via the `restraints.RST` file written during equilibration.

Output: `<project_dir>/outputs/model_0/<ligand>/repeat_<N>/`

---

### 4. Solvate and equilibrate unbound ligand

```bash
bash 04_submit_ligand_solvation_and_heating.sh <project_dir>
```

`scripts/solvate_and_heat_ligand.py` — solvates the isolated ligand (unbound stage) and runs a 6-stage equilibration:

| Stage | Process | Restraints | Notes |
|-------|---------|------------|-------|
| 01 | Minimisation | Solute | 1000 cycles |
| 02 | NVT heat 100→300 K | Solute | dt = 0.001 ps |
| 03 | NVT 300 K | Solute | dt = 0.001 ps |
| 04 | NPT | Solute | dt = 0.001 ps |
| 05 | NPT | Solute | Pos. res. weight lowered to 10 kcal/mol/Å² |
| 06 | NPT | None (free) | Final equilibrated state |

Output: `<project_dir>/equilibration/model_0/vim2/unbound/<ligand>/repeat_<N>/`

---

### 5. Build ligand transformation network

```bash
python scripts/prepare_network.py
```

Loads `model_0_sofra.json` and the set of solvated ligand PDB files, then calls `set_ligand_network` to define which ligand pairs will be transformed in the alchemical calculations.

This step will create a `lomap` directory with the network and a `png` image of the network: `<project_dir>/inputs/model_0/protein/vim2/lomap/`

---

### 6. Prepare alchemistry

```bash
bash scripts/run_alchemistry_prep.sh <project_dir> <system_name>
```

Reads the network file from `model_0_sofra.json` and submits a SLURM array job via `slurm_alchemistry_prep.sh`. Each task runs `scripts/prepare_alchemistry.py` for one ligand pair, loading the equilibrated bound (`08_free`) and unbound (`06_free`) endpoints for both ligands to set up the FEP windows.

---

## Directory layout

```
vim2/
├── ligands.txt                          # One ligand name per line (52 ligands)
├── 01_submit_solvation.sh
├── 02_submit_heat.sh
├── 03_submit_production.sh
├── 04_submit_ligand_solvation_and_heating.sh
├── slurm_alchemistry_prep.sh
└── scripts/
    ├── add_water.py                     # Bound complex solvation
    ├── heat_meze.py                     # Bound equilibration (8-stage)
    ├── production.py                    # Bound production MD
    ├── solvate_and_heat_ligand.py       # Unbound ligand solvation + equilibration
    ├── prepare_network.py               # Ligand transformation network
    ├── prepare_alchemistry.py           # RBFE window setup
    ├── solvate.sh                       # SLURM wrapper for add_water.py
    ├── heat.sh                          # SLURM wrapper for heat_meze.py
    ├── production.sh                    # SLURM wrapper for production.py
    ├── solvate_and_heat_ligand.sh       # SLURM wrapper for solvate_and_heat_ligand.py
    └── run_alchemistry_prep.sh          # Submits alchemistry prep array
```

