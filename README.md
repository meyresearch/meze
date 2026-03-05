# MetalloEnZymE parameterisation program (meze)

Metalloenzyme parameterisation tool for alchemical free energy calculations.

> [!IMPORTANT]
> This repository is in active development. The old `meze` repository can be found at https://github.com/meyresearch/metalloenzyme_rbfe/tree/main.


# Installation instructions

1. Prerequisites:

- Make sure you have `ambertools` (and `pmemd` if you want to use GPUs) installed. See [here](https://ambermd.org/Installation.php) for instructions.
- Currently the packages `meze` depends on require `cuda=12.4` if you're on a Linux/Windows machine.

3. Download `environment.yml` file:

```
curl -O https://raw.githubusercontent.com/meyresearch/meze/main/environment.yml

``` 

3. Create enivronment

```
conda env create -f environment.yml
conda activate meze-env
```
 
# Updating environment after updates to `meze`

```
conda activate meze-env
pip uninstall -y meze
pip install --no-cache-dir git+https://github.com/meyresearch/meze.git@main
```
