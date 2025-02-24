# Reproducing the Figures
| Dataset | Figure | Notebook |
|---------|-------------------|----------|
| **GLVM** | Figure 2 & S2| [`glvm/02_GLVM_paper_figures.py`](glvm/02_GLVM_paper_figures.py) |
| | Supp. Figure S1 | [`glvm/R02_GLVM_supp_figures_mask_size.py`](glvm/R02_GLVM_supp_figures_mask_size.py) |
| | Supp. Figure S3 & S4 | [`glvm/R04_GLVM_supp_figures_training_set_size.py`](glvm/R04_GLVM_supp_figures_training_set_size.py) |
| **Fly** | Figure 3 | [`fly/05_FLY_paper_figures.ipynb`](fly/05_FLY_paper_figures.ipynb) |
| **Monkey** <br> | Figure 4 | [`monkey/06_MONKEY_Decoding.ipynb`](monkey/06_MONKEY_Decoding.ipynb) |
| | Figure 5 & S5-S8| [`monkey/07_MONKEY_Encoding.ipynb`](monkey/07_MONKEY_Encoding.ipynb) |
| | Figure 6 | [`monkey/08_MONKEY_LatentAnalysis.ipynb`](monkey/08_MONKEY_LatentAnalysis.ipynb) |


> Note: Figure 1 is just a schematic and thus does not show up here. 

The datasets can be found on [Drive](https://drive.google.com/drive/folders/1Hi9xBGF2itas_agyhaQHW1yF5bvl-CHA?usp=sharing). Please load them into the data folder in the base directory or adjust the paths accordingly. 

This contains a modified format of the data provided by 
O'Doherty, J. E., Cardoso, M. M. B., Makin, J. G., & Sabes, P. N. (2017). Nonhuman Primate Reaching with Multichannel Sensorimotor Cortex Electrophysiology [Data set]. Zenodo. https://doi.org/10.5281/zenodo.583331 published under a Creative Commons Attribution 4.0 International Licencse. 


## Additional Information for running the Gaussian Latent Variable Model & Analyses 


You can either run the respective files in an **interactive window, e.g. in VS Code** or via commandline: e.g.

```
ipython 02_GLVM_paper_figures.py
```

Prior to running any of the experiments, **first generate the GLVM training, test and validation dataset**, that will then be used when running `../scripts/run_GLVM.py`.

Run the ipython files in the following order:

1. generate the dataset using `00_GLVM_generate_dataset.py`
2. start the actual training run by starting the bash script  `bash run_GLVM_many_seeds.sh` in the `bash` folder or simply via running 
`python scripts/run_GLVM.py` from the base directory.
3. aggegrate the data across many runs using `01_GLVM_data_aggregation.py`
4. Finally, you can run the paper analysis with `02_GLVM_paper_figures.py`
5. Carry out analogous steps for the supplementary analyses starting with `R_`



### Training Example
To gain an intuition for the steps required when training masked VAEs, see `03_GLVM_example_masked_VAE.ipynb`. 

In general, we advise scientists to use their VAEs they have set up for a specific dataset, train them on **all observed data** first and only once the fully observed VAE trains well, define the desired masks and start with the masking scheme. 


