# MetalloEnZymE parameterisation program (meze)

Metalloenzyme parameterisation tool for alchemical free energy calculations.

> [!IMPORTANT]
> This repository is in active development. The old `meze` repository can be found at https://github.com/meyresearch/metalloenzyme_rbfe/tree/main.


# Installation instructions

1. Download `environment.yml` file:

```
curl -O https://raw.githubusercontent.com/meyresearch/meze/main/environment.yml

``` 

2. Create enivronment

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
