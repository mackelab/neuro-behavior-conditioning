# Modeling conditional distributions of neural and behavioral data with masked variational autoencoders

### Purpose of this repostitory

This repository contains research code for the [preprint](https://www.biorxiv.org/content/10.1101/2024.04.19.590082v1):   
 ***Modeling conditional distributions of neural and behavioral data with masked variational autoencoders***   
 by Schulz, Vetter, Gao, Morales, Lobato-Rios, Ramdya, Goncalves*, and Macke* (2024).



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
cd notebooks
python 00_GLVM_generate_dataset.py 
```

The python scripts to run the autoencoder training can be started directly from the base directory. Experiment tracking and saving with [Weights & Biases](https://wandb.ai/site) is supported, but disabled by default.

```
python scripts/run_GLVM.py
```

To start multiple runs with different seeds run
```
cd bash
bash run_GLVM_many_seeds.sh
```

See the notebooks [README](./notebooks/README.md) for further instructions for postprocessing and plotting of the results.

### Fly walking behavior

We made our dataset of walking behavior of flies (Drosophila melanogaster) publicly available at 
https://zenodo.org/records/11002776. Please download it from there and store it in `./data/fly/`. Please ensure the download was successful using `04_FLY_read_in_data.ipynb`

The python scripts to run the autoencoder training on the fly walking dataset can be called in the same way (coming soon!):
```
python scripts/run_fly.py
```
> Note: training the autoencoder for fly walking behavior will take substantially longer and training on a  `GPU` is recommended.

### Monkey reach task 

The monkey reach task uses data kindly shared by O'Doherty, Cardoso, Maki and Sabes at https://zenodo.org/records/583331. For the results in the preprint, we trained only on one session `loco_20170213_02`. The scripts for training and evalution on this task will come soon.


> Note that this repository is work in progress. 

## Citation

```
@article{schulz_2024_conditional,
	author = {Auguste Schulz and Julius Vetter and Richard Gao and Daniel Morales and Victor Lobato-Rios and Pavan Ramdya and Pedro J. Gon{\c{c}}alves and Jakob H. Macke},  
	title = {Modeling conditional distributions of neural and behavioral data with masked variational autoencoders},  
	year = {2024},  
	doi = {10.1101/2024.04.19.590082},  
	publisher = {Cold Spring Harbor Laboratory},  
	journal = {bioRxiv}  
}
```
## Contact
Questions regarding the code should be addressed to auguste.schulz@uni-tuebingen.de.
