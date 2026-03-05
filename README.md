# MetalloEnZymE parameterisation program (meze)

Metalloenzyme parameterisation tool for alchemical free energy calculations.

> [!IMPORTANT]
> This repository is in active development. The old `meze` repository can be found at https://github.com/meyresearch/metalloenzyme_rbfe/tree/main.


# Installation instructions

1. Prerequisites:

- Make sure you have `ambertools` (and `pmemd` if you want to use GPUs) installed. See [here](https://ambermd.org/Installation.php) for instructions.
- Currently the packages `meze` depends on require `cuda=12.4` if you're on a Linux/Windows machine.
- Ideally, you should have `miniforge3` so that you can install `meze` with `mamba`. The installation will work with `conda` as well, but will be significantly slower. See [here](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install) for installation instructions.

3. Download `environment.yml` file:

```
curl -O https://raw.githubusercontent.com/meyresearch/meze/main/environment.yml

``` 

3. Create enivronment

```
mamba env create -f environment.yml
mamba activate meze-env
```
 
# Updating environment after updates to `meze`

```
mamba activate meze-env
pip uninstall -y meze
pip install --no-cache-dir git+https://github.com/meyresearch/meze.git@main
```

If there are updates to the `environment.yml`:

First, get the latest `environment.yml`:
```
curl -O https://raw.githubusercontent.com/meyresearch/meze/main/environment.yml
```

Then:

```
mamba env update -n meze-env -f environment.yml --prune
mamba activate meze-env
pip uninstall -y meze
pip install --no-cache-dir git+https://github.com/meyresearch/meze.git@main
```
