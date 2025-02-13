# scCMA

scCMA: a constrasive masked autoencoder for single-cell RNA-seq clustering

## Datasets

Place the h5 dataset in the `data/` directory,
h5 file contains gene expression X and true label Y.

## Usage

Set the dataset read path and result storage path

```Python
# Datasets directory and output directory
args["paths"] = {"data": "./data/", "results": "./result/"} 
```
Parameter settings.

```Python
hidden_size=128,
dropout=0.1,
masked_data_weight=0.75,
mask_loss_weight=0.7,
contrastive_weight=0.07
```

scCMA can be run with the following commands.

```shell
python run_scCMA.py
```