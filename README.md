# Modeling conditional distributions of neural and behavioral data with masked variational autoencoders

### Purpose of this repostitory

This repository contains research code for the [paper](https://www.cell.com/cell-reports/abstract/S2211-1247(25)00109-3):   
 ***Modeling conditional distributions of neural and behavioral data with masked variational autoencoders***  
 by Schulz, Vetter, Gao, Morales, Lobato-Rios, Ramdya, Goncalves*, and Macke*. Cell Reports (2025).



## Installation

To run the scripts make sure to first install all the requirements. We recommend creating a conda environment first.
A `GPU` is recommend but not necessary.



``` 
git clone git@github.com:mackelab/neuro-behavior-conditioning.git
cd neuro-behavior-conditioning
conda create --name nbc python=3.8
conda activate nbc
pip install -e .
```

Make sure that the `Jupiter` package is also installed.

### Gaussian Latent Variable Model

Before running the training script for the simluated dataset (Gaussian Latent Variable Model), you need to generate the dataset which will be stored in `./data/glvm/`:

```
cd notebooks/glvm/
ipython 00_GLVM_generate_dataset.py 
```


### Fly walking behavior

We made our dataset of walking behavior of flies (Drosophila melanogaster) publicly available at 
https://zenodo.org/records/11002776. Please download it from there and store it in `./data/fly/`. Please ensure the download was successful using [04_FLY_read_in_data.ipynb](./notebooks/fly/04_FLY_read_in_data.ipynb) in `notebooks/fly/`

### Monkey reach task 

The monkey reach task uses data kindly shared by O'Doherty, Cardoso, Maki and Sabes at https://zenodo.org/records/583331. For the results in the preprint, we trained only on one session `loco_20170213_02`. 



## Training the models 

The python scripts to run the autoencoder training can be started directly from the base directory. Experiment tracking and saving with [Weights & Biases](https://wandb.ai/site) is supported, but disabled by default. E.g. for the GLVM dataset run:

```
python scripts/run_GLVM.py
```

To start multiple runs with different seeds run
```
cd bash
bash run_GLVM_many_seeds.sh
```

See the additional [notebooks README](./notebooks/README.md) for further instructions for postprocessing, plotting of the results and reproducing the respective paper figures.

The added letter 'R_' in the file names, indicates differences between the [preprint](https://www.biorxiv.org/content/10.1101/2024.04.19.590082v1)  and the [published](https://www.cell.com/cell-reports/abstract/S2211-1247(25)00109-3) version of this manuscript. 

## Citation

```
@article{schulz_2025_conditional,
	title = {Modeling conditional distributions of neural and behavioral data with masked variational autoencoders},  
	author = {Schulz, Auguste and Vetter, Julius and Gao, Richard and Morales, Daniel and Lobato-Rios, Victor and Ramdya, Pavan and Gon{\c{c}}alves, Pedro J. and Macke, Jakob H.},  
	journal = {Cell Reports},  
	year = {2025},
	volume = {44},
	number = {3},
	url = {https://www.cell.com/cell-reports/abstract/S2211-1247(25)00109-3},
	doi = {10.1016/j.celrep.2025.115338},
}
```
## Contact
Questions regarding the code should be addressed to auguste.schulz@uni-tuebingen.de.
