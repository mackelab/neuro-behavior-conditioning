## Gaussian Latent Variable Model runs 


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



### Training Example



To gain an intuition for the steps required when training masked VAEs, see `03_GLVM_example_masked_VAE.ipynb`. 



## Monkey reach task and fly walking behavior

> Note: Scripts for training the VAEs and generating the figures and download links to the data will come soon! 
